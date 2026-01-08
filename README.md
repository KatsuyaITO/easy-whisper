# Easy Whisper

A simple web application for audio transcription using OpenAI's Whisper model. This application provides a user-friendly web interface for transcribing audio files using the powerful Whisper speech recognition model.

## Features

- **User-friendly Web Interface**: Easy-to-use interface for audio file upload and transcription
- **Dynamic Model Selection**: Choose from 11 different Whisper models directly in the UI
  - Multilingual models (tiny, base, small, medium, large, large-v3, large-v3-turbo)
  - English-only optimized models (tiny.en, base.en, small.en, medium.en)
- **Flexible Device Selection**: Choose between auto-detect, CUDA (GPU), or CPU processing
- **Custom Output Filenames**: Specify your own filename for downloaded transcriptions
- **Automatic Audio Format Conversion**: Uses ffmpeg to handle various audio formats
- **GPU Acceleration**: CUDA support for faster transcription when available
- **Model Caching**: Intelligent caching system to reuse loaded models
- **Real-time Progress Feedback**: Visual loading indicators during transcription
- **Easy Export**: Copy to clipboard or download as text file with custom naming

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
| `PORT` | `7860` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Available Whisper Models

You can now select models directly in the web interface, or set a default by configuring `WHISPER_MODEL_ID` in your `.env` file:

**Multilingual Models** (support 99+ languages):
- `openai/whisper-tiny` - Fastest, least accurate (~39M parameters)
- `openai/whisper-base` - Fast, basic accuracy (~74M parameters)
- `openai/whisper-small` - Balanced speed/accuracy (~244M parameters)
- `openai/whisper-medium` - Good accuracy, slower (~769M parameters)
- `openai/whisper-large` - High accuracy, slower (~1550M parameters)
- `openai/whisper-large-v3` - Best accuracy, slowest (default, ~1550M parameters)
- `openai/whisper-large-v3-turbo` - Fastest large model with excellent accuracy (~809M parameters)

**English-Only Models** (optimized for English):
- `openai/whisper-tiny.en` - Fastest English-only model
- `openai/whisper-base.en` - Fast English-only model
- `openai/whisper-small.en` - Balanced English-only model
- `openai/whisper-medium.en` - High accuracy English-only model

## Usage

1. Start the application (it will be available at http://localhost:7860 by default)
2. Open your web browser and navigate to the application URL
3. Configure your transcription settings in the web interface:
   - **Select Whisper Model**: Choose from 11 different models based on your accuracy/speed needs
   - **Select Processing Device**: Choose auto-detect, CUDA (GPU), or CPU
   - **Set Output Filename** (optional): Customize the name for your transcription file
4. Upload an audio file using the drag-and-drop interface or file browser
5. Click "Transcribe Audio" and wait for the transcription to complete
6. Review the transcription result
7. Copy to clipboard or download the transcription with your custom filename

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
uv run uvicorn easy_whisper:app --reload --host localhost --port 7860
```

### New UI Features

The application now supports dynamic configuration through the web interface:

- **Model Selection**: Users can select from 11 different Whisper models without needing to modify environment variables
- **Device Selection**: Choose between auto-detect, CUDA, or CPU processing on a per-request basis
- **Custom Filenames**: Set custom output filenames for downloaded transcriptions
- **Model Caching**: The backend intelligently caches loaded models to improve performance when switching between models

### Project Structure

```
easy-whisper/
├── easy_whisper/          # Main package
│   ├── __init__.py        # Main application code with FastAPI app
│   └── templates/         # HTML templates
│       └── index.html     # Web interface with dynamic controls
├── pyproject.toml         # Project dependencies and metadata
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore rules
└── README.md              # This file
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
