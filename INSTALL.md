# Installation Guide for Data Science Toolkit

This guide will help you install the Data Science Toolkit and resolve common dependency issues.

## Quick Installation (Recommended)

For most users, the core installation without database drivers is recommended:

```bash
pip install -r requirements-core.txt
```

This installs all essential dependencies without system-level requirements.

## Dealing with System Dependencies

Some optional packages require system-level libraries. Here's how to handle them:

### 1. MySQL Support (mysqlclient)

The error you encountered is because `mysqlclient` requires MySQL development libraries.

**On Ubuntu/Debian:**

```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install pkg-config
sudo apt-get install python3-dev default-libmysqlclient-dev build-essential

# Then install the Python package
pip install mysqlclient
```

**On macOS:**

```bash
# Using Homebrew
brew install mysql pkg-config

# Then install the Python package
pip install mysqlclient
```

**On Windows:**

```bash
# Download the wheel file from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysqlclient
# Then install it:
pip install mysqlclient‑1.4.6‑cp39‑cp39‑win_amd64.whl
```

**Alternative: Use PyMySQL (pure Python, no system deps):**

```bash
pip install pymysql
```

### 2. PostgreSQL Support (psycopg2)

**Recommended: Use the binary version (no system deps needed):**

```bash
pip install psycopg2-binary
```

**Or install system dependencies:**

```bash
# Ubuntu/Debian
sudo apt-get install libpq-dev

# macOS
brew install postgresql

# Then install
pip install psycopg2
```

### 3. Additional System Dependencies

**For Ubuntu/Debian systems, install common development tools:**

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    pkg-config \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    libagg-dev \
    libhdf5-dev \
    libnetcdf-dev \
    gfortran \
    libatlas-base-dev \
    liblapack-dev
```

## Installation Options

### 1. Minimal Installation

```bash
# Only the absolute essentials
pip install numpy pandas scikit-learn matplotlib
```

### 2. Core Installation (Recommended)

```bash
pip install -r requirements-core.txt
```

### 3. Full Installation (Advanced Users)

```bash
# First install system dependencies (see above)
# Then install Python packages
pip install -r requirements.txt
```

### 4. Development Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd data-science-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements-core.txt
pip install -e .
```

## Troubleshooting Common Issues

### 1. Memory Issues During Installation

Some packages (especially deep learning libraries) are large. If you encounter memory issues:

```bash
# Install packages one by one
pip install numpy
pip install pandas
pip install scikit-learn
# ... and so on
```

### 2. Conflicting Dependencies

If you encounter dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install -r requirements-core.txt
```

### 3. GPU Support for Deep Learning

**For PyTorch with CUDA:**

```bash
# Check your CUDA version first
nvidia-smi

# Install appropriate PyTorch version
# For CUDA 11.3
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# For CPU only
pip install torch torchvision torchaudio
```

**For TensorFlow with GPU:**

```bash
# TensorFlow 2.x includes GPU support by default
pip install tensorflow

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 4. Platform-Specific Issues

**Windows:**

- Some packages may require Visual C++ 14.0 or greater. Install from:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/

**macOS:**

- Ensure Xcode Command Line Tools are installed:
  ```bash
  xcode-select --install
  ```

**Linux:**

- Most distributions work out of the box, but ensure Python development headers are installed:
  ```bash
  sudo apt-get install python3-dev  # Debian/Ubuntu
  sudo yum install python3-devel     # RedHat/CentOS
  ```

## Verifying Installation

After installation, verify everything is working:

```python
# test_installation.py
import sys
print(f"Python version: {sys.version}")

try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
except ImportError:
    print("NumPy not installed")

try:
    import pandas
    print(f"Pandas version: {pandas.__version__}")
except ImportError:
    print("Pandas not installed")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("Scikit-learn not installed")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("PyTorch not installed")

print("\nInstallation verification complete!")
```

## Docker Alternative

For a hassle-free installation, use Docker:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-core.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-core.txt

# Copy your code
COPY . .

CMD ["python"]
```

Build and run:

```bash
docker build -t data-science-toolkit .
docker run -it data-science-toolkit
```

## Getting Help

If you continue to have issues:

1. Check the specific package's documentation
2. Search for the error message online
3. Create an issue in the repository with:
   - Your operating system
   - Python version
   - The complete error message
   - Steps you've already tried

Remember, you don't need all packages to get started. Begin with `requirements-core.txt` and add additional packages as needed for your specific use case.
