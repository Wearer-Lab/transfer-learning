# Transfer Learning Video Processing Pipeline

A powerful CLI tool for processing videos and generating step-by-step guides using AI. This tool can handle both local video files and YouTube videos, extracting key frames and generating detailed guides using OpenAI's GPT-4 Vision API.

## Features

- Process local video files
- Download and process YouTube videos
- Extract frames at specified intervals
- Generate step-by-step guides with AI analysis
- Beautiful CLI interface with progress tracking
- Support for both local and YouTube video sources
- Built-in monitoring and metrics collection
- Optimized performance with async processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Wearer-Lab/transfer-learning.git
cd transfer-learning
```

2. Create a virtual environment and install dependencies using UV:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install uv
uv pip install -e .
```

3. Set up your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Usage

The CLI provides several commands for different operations:

### Process a Local Video

Extract frames from a local video file:

```bash
transfer-learning process-video path/to/video.mp4 --output-dir output --batch-size 30
```

### Process a YouTube Video

Download and process a YouTube video:

```bash
transfer-learning process-youtube "https://youtube.com/watch?v=VIDEO_ID" --output-dir output
```

### Generate a Guide

Generate a step-by-step guide from processed video frames:

```bash
transfer-learning generate-guide output/processed_data --output-dir guides
```

### Complete YouTube Pipeline

Download a YouTube video and generate a guide in one command:

```bash
transfer-learning youtube-guide "https://youtube.com/watch?v=VIDEO_ID" --output-dir output
```

### Transcribe Audio

Transcribe audio from a video file:

```bash
transfer-learning transcribe path/to/video.mp4 --output-dir transcripts
```

### Analyze Video Content

Analyze video content to extract key information:

```bash
transfer-learning analyze path/to/video.mp4 --output-dir analysis
```

### Download Video

Download a video from YouTube or other supported platforms:

```bash
transfer-learning download "https://youtube.com/watch?v=VIDEO_ID" --output-dir videos
```

## Command Options

All commands support various options. Use the `--help` flag to see available options for each command:

```bash
transfer-learning process-video --help
```

## Monitoring and Metrics

Transfer Learning includes built-in monitoring capabilities:

- **Metrics Collection**: Performance metrics are collected during processing and saved to the `metrics` directory
- **Logging**: Detailed logs are saved to the `logs` directory
- **Progress Tracking**: Rich progress bars and status indicators in the terminal

To view collected metrics:

```bash
transfer-learning config --show-metrics
```

## Output Format

The generated guide is saved as a JSON file with the following structure:

```json
{
  "title": "Process Title",
  "description": "Overall process description",
  "steps": [
    {
      "step_number": 1,
      "title": "Step Title",
      "description": "Step description",
      "tools_used": ["tool1", "tool2"],
      "duration": 10.5,
      "key_points": ["point1", "point2"],
      "timestamp": 0.0
    }
  ],
  "principles": [
    {
      "name": "Principle Name",
      "description": "Principle description",
      "importance": "Why this principle matters",
      "examples": ["example1", "example2"]
    }
  ],
  "total_duration": 120.5,
  "source_type": "local",
  "source_path": "path/to/video.mp4"
}
```

## Development

To set up the development environment:

1. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Check code style:
```bash
ruff check .
```

## Project Structure

```
transfer-learning/
├── data/               # Data storage directory
├── docs/               # Documentation
├── logs/               # Log files
├── metrics/            # Metrics collection
├── src/                # Source code
│   └── transfer_learning/
│       ├── core/       # Core processing modules
│       ├── guide/      # Guide generation
│       ├── models/     # AI model interfaces
│       ├── monitoring/ # Monitoring and metrics
│       ├── utils/      # Utility functions
│       ├── cli.py      # CLI implementation
│       └── config.py   # Configuration
├── tests/              # Test suite
├── .env                # Environment variables
├── pyproject.toml      # Project configuration
└── setup.py            # Package setup
```

## License

This project is licensed under the Eclipse Public License - v 2.0 - see the LICENSE file for details. 