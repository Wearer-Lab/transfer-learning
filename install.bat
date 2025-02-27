@echo off
:: Transfer Learning Installation Script for Windows
:: This script installs and sets up the Transfer Learning video processing pipeline

echo [94m>> Starting Transfer Learning installation...[0m

:: Check if Python 3.9+ is installed
echo [94m>> Checking Python version...[0m
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [91m>> Python not found. Please install Python 3.9 or higher[0m
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo [91m>> Python 3.9 or higher is required, but %PYTHON_VERSION% was found[0m
    exit /b 1
)

if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 9 (
        echo [91m>> Python 3.9 or higher is required, but %PYTHON_VERSION% was found[0m
        exit /b 1
    )
)

echo [92m>> Python %PYTHON_VERSION% detected[0m

:: Create virtual environment
echo [94m>> Creating virtual environment...[0m
if exist .venv (
    echo [94m>> Virtual environment already exists[0m
) else (
    python -m venv .venv
    echo [92m>> Virtual environment created[0m
)

:: Activate virtual environment
echo [94m>> Activating virtual environment...[0m
call .venv\Scripts\activate
echo [92m>> Virtual environment activated[0m

:: Install UV package manager
echo [94m>> Installing UV package manager...[0m
pip show uv > nul 2>&1
if %ERRORLEVEL% neq 0 (
    pip install uv
    echo [92m>> UV installed[0m
) else (
    echo [94m>> UV already installed[0m
)

:: Install dependencies
echo [94m>> Installing dependencies...[0m
uv pip install -e .
echo [92m>> Dependencies installed[0m

:: Ask if user wants to install development dependencies
echo.
set /p INSTALL_DEV="Do you want to install development dependencies? (y/n): "
if /i "%INSTALL_DEV%"=="y" (
    echo [94m>> Installing development dependencies...[0m
    uv pip install -e ".[dev]"
    echo [92m>> Development dependencies installed[0m
)

:: Create directories
echo [94m>> Creating directories...[0m
if not exist data\videos mkdir data\videos
if not exist data\frames mkdir data\frames
if not exist data\transcripts mkdir data\transcripts
if not exist data\guides mkdir data\guides
if not exist data\analysis mkdir data\analysis
if not exist data\temp mkdir data\temp
if not exist logs mkdir logs
if not exist metrics mkdir metrics
if not exist .cache mkdir .cache
echo [92m>> Directories created[0m

:: Create .env file if it doesn't exist
echo [94m>> Setting up environment variables...[0m
if exist .env (
    echo [94m>> .env file already exists[0m
) else (
    echo # Transfer Learning Configuration > .env
    echo. >> .env
    echo # API Keys >> .env
    echo OPENAI_API_KEY=your-api-key-here >> .env
    echo # ANTHROPIC_API_KEY=your-api-key-here >> .env
    echo # HUGGINGFACE_API_KEY=your-api-key-here >> .env
    echo. >> .env
    echo # Processing Configuration >> .env
    echo FRAME_EXTRACTION_INTERVAL=30 >> .env
    echo MAX_FRAMES_PER_VIDEO=100 >> .env
    echo BATCH_SIZE=30 >> .env
    echo MAX_CONCURRENT_BATCHES=100 >> .env
    echo. >> .env
    echo # Model Configuration >> .env
    echo OPENAI_MODEL=gpt-4o-mini >> .env
    echo VISION_MODEL=o3-mini >> .env
    echo WHISPER_MODEL=base >> .env
    echo WHISPER_DEVICE=cpu >> .env
    echo. >> .env
    echo # Monitoring Configuration >> .env
    echo ENABLE_MONITORING=true >> .env
    echo LOG_LEVEL=INFO >> .env
    echo METRICS_ENABLED=true >> .env
    echo. >> .env
    echo # Cache Configuration >> .env
    echo ENABLE_CACHE=true >> .env
    echo CACHE_TTL_HOURS=24 >> .env
    echo [92m>> .env file created[0m
    echo [94m>> Please edit the .env file to add your API keys[0m
)

echo [92m>> Installation complete![0m
echo [94m>> To activate the virtual environment, run:[0m
echo     .venv\Scripts\activate
echo [94m>> To get started, run:[0m
echo     transfer-learning --help
echo [94m>> Don't forget to add your OpenAI API key to the .env file![0m

pause 