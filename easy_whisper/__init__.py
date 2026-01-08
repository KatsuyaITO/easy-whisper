import logging
import os
import subprocess
import traceback
import uuid
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper Transcription App")

# Get templates directory relative to this file (inside package)
_package_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(_package_dir / "templates"))

# Use absolute path for uploads (default to ./uploads in current working directory)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).absolute()
UPLOAD_DIR.mkdir(exist_ok=True)

# Whisper model configuration
DEFAULT_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "openai/whisper-large-v3")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache for pipelines (key: model_id + device)
pipeline_cache = {}


def get_pipeline(model_id: str = None, device: str = "auto"):
    """Load Whisper pipeline with caching support."""
    # Use defaults if not specified
    if model_id is None:
        model_id = DEFAULT_MODEL_ID

    # Handle auto device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Create cache key
    cache_key = f"{model_id}:{device}"

    # Return cached pipeline if available
    if cache_key in pipeline_cache:
        logger.info(f"Using cached pipeline for {cache_key}")
        return pipeline_cache[cache_key]

    # Load new pipeline
    logger.info(f"Loading Whisper model {model_id} on {device}...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Cache the pipeline
    pipeline_cache[cache_key] = pipe
    logger.info(f"Model loaded successfully! Cache key: {cache_key}")

    return pipe


def convert_to_wav(input_path: Path) -> Path:
    """Convert audio file to WAV format (16kHz mono) for Whisper using ffmpeg."""
    output_path = input_path.with_suffix(".wav")
    logger.info(f"Converting {input_path} to WAV format...")

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",       # mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg conversion failed: {result.stderr}")
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    logger.info(f"Conversion successful: {output_path}")
    return output_path


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL_ID),
    device: str = Form("auto"),
    filename: str = Form("")
):
    """Transcribe uploaded audio file."""
    logger.info(f"Received transcription request for file: {file.filename}")
    logger.info(f"Model: {model}, Device: {device}, Custom filename: {filename}")
    temp_path = None
    wav_path = None

    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        original_ext = Path(file.filename).suffix or ".audio"
        temp_path = UPLOAD_DIR / f"{file_id}{original_ext}"
        logger.info(f"Saving uploaded file to: {temp_path}")

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"File saved successfully, size: {len(content)} bytes")

        # Convert to WAV format
        wav_path = convert_to_wav(temp_path)

        # Load pipeline with user-selected model and device
        logger.info(f"Loading transcription pipeline for model: {model} on device: {device}...")
        transcription_pipe = get_pipeline(model_id=model, device=device)

        logger.info(f"Starting transcription of: {wav_path}")
        result = transcription_pipe(
            str(wav_path),
            return_timestamps=True,
            generate_kwargs={"language": None, "task": "transcribe"},
        )
        logger.info("Transcription completed successfully")

        # Clean up files
        temp_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        logger.info("Temporary files cleaned up")

        text = result["text"].strip()
        logger.info(f"Transcription result length: {len(text)} characters")

        # Return HTML partial for HTMX
        return HTMLResponse(f"""
        <div class="result-container">
            <h3>Transcription Result</h3>
            <div class="transcription-text" id="transcription-output">{text}</div>
            <div class="button-group">
                <button onclick="copyToClipboard()" class="btn btn-copy">Copy to Clipboard</button>
                <button onclick="downloadText()" class="btn btn-download">Download as Text</button>
            </div>
        </div>
        """)

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Clean up files on error
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if wav_path and wav_path.exists():
            wav_path.unlink(missing_ok=True)

        return HTMLResponse(f"""
        <div class="error">
            <h3>Error</h3>
            <p>{str(e)}</p>
        </div>
        """, status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "default_device": DEFAULT_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "default_model": DEFAULT_MODEL_ID,
        "cached_pipelines": len(pipeline_cache),
        "cache_keys": list(pipeline_cache.keys())
    }


def main():
    """Main entry point for the application."""
    import uvicorn
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
