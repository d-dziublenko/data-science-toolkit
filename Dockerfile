# Multi-stage Dockerfile for Universal Data Science Toolkit
# This Dockerfile creates an optimized container for running data science workloads

# Stage 1: Base image with system dependencies
FROM python:3.9-slim as base

# Set environment variables to optimize Python in container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for data science libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials for compiling Python packages
    build-essential \
    gcc \
    g++ \
    # Scientific computing libraries
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    # HDF5 for data storage
    libhdf5-dev \
    # Graphics libraries for visualization
    libfreetype6-dev \
    libpng-dev \
    # Database clients
    libpq-dev \
    # Useful utilities
    curl \
    wget \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies builder
FROM base as builder

# Create virtual environment for clean dependency management
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements files first (for better caching)
WORKDIR /tmp
COPY requirements*.txt ./

# Install dependencies in order of importance
# Core dependencies first (most stable, less likely to change)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-core.txt

# Optional dependencies (comment out if not needed)
# RUN pip install -r requirements-full.txt

# Stage 3: Final runtime image
FROM base as runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash dsuser

# Set up working directory
WORKDIR /app

# Copy application code
COPY --chown=dsuser:dsuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/data/external \
             /app/outputs /app/logs /app/models /app/configs && \
    chown -R dsuser:dsuser /app

# Switch to non-root user
USER dsuser

# Expose ports for Jupyter and API services
EXPOSE 8888 8000

# Health check to ensure container is running properly
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import pandas, numpy, sklearn; print('Health check passed')" || exit 1

# Default command: Start Jupyter Lab
# Can be overridden with docker run command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Alternative commands (uncomment one):
# CMD ["python", "-m", "pipelines.training", "--config", "/app/configs/default.yaml"]
# CMD ["python", "-m", "utils.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["/bin/bash"]

# Stage 4: Development image (optional, for development work)
FROM runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Development utilities
    less \
    htop \
    tmux \
    # Code quality tools
    black \
    flake8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt

USER dsuser

# Override CMD for development
CMD ["/bin/bash"]

# Build instructions:
# - Production image: docker build -t ds-toolkit:latest --target runtime .
# - Development image: docker build -t ds-toolkit:dev --target development .
# - With full dependencies: docker build -t ds-toolkit:full --build-arg INSTALL_FULL=true .