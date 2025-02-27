from setuptools import setup, find_packages

setup(
    name="transfer_learning",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.7.0",
        "opencv-python>=4.8.0",
        "yt-dlp>=2023.12.30",
        "openai>=1.12.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.6.1",
        "pydantic-settings>=2.1.0",
        "psutil>=5.9.0",
        "moviepy>=2.0.0",
        "whisper>=1.1.10",
        "faster-whisper>=0.10.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
            "black>=23.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transfer-learning=transfer_learning.cli:main",
        ],
    },
) 