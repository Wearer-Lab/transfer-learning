 #!/bin/bash

# Transfer Learning Installation Script
# This script installs and sets up the Transfer Learning video processing pipeline

set -e

# Print colored messages
print_message() {
    echo -e "\033[1;34m>> $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m>> $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m>> $1\033[0m"
}

# Check if Python 3.9+ is installed
check_python() {
    print_message "Checking Python version..."
    if command -v python3 &>/dev/null; then
        python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ $(echo "$python_version >= 3.9" | bc) -eq 1 ]]; then
            print_success "Python $python_version detected"
        else
            print_error "Python 3.9 or higher is required, but $python_version was found"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9 or higher"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_message "Creating virtual environment..."
    if [ -d ".venv" ]; then
        print_message "Virtual environment already exists"
    else
        python3 -m venv .venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    print_success "Virtual environment activated"
}

# Install UV package manager
install_uv() {
    print_message "Installing UV package manager..."
    if command -v uv &>/dev/null; then
        print_message "UV already installed"
    else
        pip install uv
        print_success "UV installed"
    fi
}

# Install dependencies
install_dependencies() {
    print_message "Installing dependencies..."
    uv pip install -e .
    print_success "Dependencies installed"
}

# Install development dependencies
install_dev_dependencies() {
    print_message "Installing development dependencies..."
    uv pip install -e ".[dev]"
    print_success "Development dependencies installed"
}

# Create directories
create_directories() {
    print_message "Creating directories..."
    mkdir -p data/videos data/frames data/transcripts data/guides data/analysis data/temp
    mkdir -p logs
    mkdir -p metrics
    mkdir -p .cache
    print_success "Directories created"
}

# Create .env file if it doesn't exist
create_env_file() {
    print_message "Setting up environment variables..."
    if [ -f ".env" ]; then
        print_message ".env file already exists"
    else
        cat > .env << EOF
# Transfer Learning Configuration

# API Keys
OPENAI_API_KEY=your-api-key-here
# ANTHROPIC_API_KEY=your-api-key-here
# HUGGINGFACE_API_KEY=your-api-key-here

# Processing Configuration
FRAME_EXTRACTION_INTERVAL=30
MAX_FRAMES_PER_VIDEO=100
BATCH_SIZE=30
MAX_CONCURRENT_BATCHES=100

# Model Configuration
OPENAI_MODEL=gpt-4o-mini
VISION_MODEL=o3-mini
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Monitoring Configuration
ENABLE_MONITORING=true
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Cache Configuration
ENABLE_CACHE=true
CACHE_TTL_HOURS=24
EOF
        print_success ".env file created"
        print_message "Please edit the .env file to add your API keys"
    fi
}

# Run the installation
main() {
    print_message "Starting Transfer Learning installation..."
    
    check_python
    create_venv
    install_uv
    install_dependencies
    
    # Ask if user wants to install development dependencies
    read -p "Do you want to install development dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_dev_dependencies
    fi
    
    create_directories
    create_env_file
    
    print_success "Installation complete!"
    print_message "To activate the virtual environment, run:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "    .venv\\Scripts\\activate"
    else
        echo "    source .venv/bin/activate"
    fi
    print_message "To get started, run:"
    echo "    transfer-learning --help"
    print_message "Don't forget to add your OpenAI API key to the .env file!"
}

main