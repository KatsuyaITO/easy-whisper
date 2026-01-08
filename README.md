# Easy Whisper

A simple web application for audio transcription using OpenAI's Whisper model. This application provides a user-friendly web interface for transcribing audio files using the powerful Whisper speech recognition model.

## Features

- Web-based interface for easy audio file upload
- Automatic audio format conversion using ffmpeg
- GPU acceleration support (CUDA) when available
- Support for multiple Whisper model sizes
- Configurable through environment variables
- Real-time transcription with progress feedback
- Copy to clipboard and download transcription results

## Prerequisites

- Python 3.13 or higher
- ffmpeg (for audio conversion)
- CUDA-compatible GPU (optional, for faster transcription)

### Installing ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Quick Start

### Option 1: Run with uvx (Recommended)

The easiest way to run Easy Whisper is using `uvx`:

```bash
uvx --from git+https://github.com/KatsuyaITO/easy-whisper easy-whisper
```

This will automatically install all dependencies and start the application.

### Option 2: Install with uv

```bash
# Clone the repository
git clone https://github.com/KatsuyaITO/easy-whisper.git
cd easy-whisper

# Run with uv
uv run easy-whisper
```

### Option 3: Traditional Installation

```bash
# Clone the repository
git clone https://github.com/KatsuyaITO/easy-whisper.git
cd easy-whisper

# Install dependencies
uv sync

# Run the application
uv run python main.py
```

## Configuration

Easy Whisper can be configured using environment variables. Copy the example file and customize it:

```bash
cp .env.example .env
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_ID` | `openai/whisper-large-v3` | Whisper model to use |
| `UPLOAD_DIR` | `uploads` | Directory for temporary file uploads |
| `HOST` | `localhost` | Server host address |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Available Whisper Models

You can change the model by setting `WHISPER_MODEL_ID` in your `.env` file:

- `openai/whisper-tiny` - Fastest, least accurate
- `openai/whisper-base` - Fast, basic accuracy
- `openai/whisper-small` - Balanced speed/accuracy
- `openai/whisper-medium` - Good accuracy, slower
- `openai/whisper-large-v3` - Best accuracy, slowest (default)

## Usage

1. Start the application (it will be available at http://localhost:8000 by default)
2. Open your web browser and navigate to the application URL
3. Upload an audio file using the web interface
4. Wait for the transcription to complete
5. Copy or download the transcription result

## API Endpoints

- `GET /` - Web interface
- `POST /transcribe` - Upload and transcribe audio file
- `GET /health` - Health check endpoint

## Development

### Running in Development Mode

```bash
# Install development dependencies
uv sync

# Run with auto-reload
uv run uvicorn main:app --reload --host localhost --port 8000
```

### Project Structure

```
easy-whisper/
├── main.py              # Main application code
├── templates/           # HTML templates
│   └── index.html      # Web interface
├── pyproject.toml      # Project dependencies and metadata
├── .env.example        # Example environment variables
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors, try:
1. Using a smaller model (e.g., `openai/whisper-small`)
2. Running on CPU (the application will automatically fallback to CPU if CUDA is unavailable)

### ffmpeg Not Found

Ensure ffmpeg is installed and available in your system PATH. Verify with:
```bash
ffmpeg -version
```

### Model Download Issues

On first run, the application will download the Whisper model. This may take some time depending on your internet connection. The models are cached locally for subsequent runs.

## License

This project is open source and available for personal and educational use.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for the model implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
