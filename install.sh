#!/bin/bash

# ============================================================================
# Universal Data Science Toolkit - Installation Script
# ============================================================================
# This script automates the installation process for the Data Science Toolkit
# It handles environment setup, dependency installation, and verification
# ============================================================================

set -e  # Exit on error

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
PROJECT_NAME="Universal Data Science Toolkit"
PYTHON_MIN_VERSION="3.8"
PYTHON_RECOMMENDED="3.9"
VENV_NAME="venv"
INSTALL_DIR=$(pwd)

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare version numbers
version_gt() {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"
}

# ASCII art banner
print_banner() {
    echo "======================================================================="
    echo "   ____        _          ____       _                      "
    echo "  |  _ \  __ _| |_ __ _  / ___|  ___(_) ___ _ __   ___ ___ "
    echo "  | | | |/ _\` | __/ _\` | \___ \ / __| |/ _ \ '_ \ / __/ _ \\"
    echo "  | |_| | (_| | || (_| |  ___) | (__| |  __/ | | | (_|  __/"
    echo "  |____/ \__,_|\__\__,_| |____/ \___|_|\___|_| |_|\___\___|"
    echo "                                                            "
    echo "                    T O O L K I T                          "
    echo "======================================================================="
    echo ""
}

# Function to check system requirements
check_system_requirements() {
    print_info "Checking system requirements..."
    
    # Check OS
    OS=$(uname -s)
    case "$OS" in
        Linux*)     OS_TYPE="Linux";;
        Darwin*)    OS_TYPE="Mac";;
        CYGWIN*|MINGW*|MSYS*) OS_TYPE="Windows";;
        *)          OS_TYPE="Unknown";;
    esac
    
    print_info "Detected OS: $OS_TYPE"
    
    # Check Python installation
    if ! command_exists python3; then
        print_error "Python 3 is not installed!"
        echo "Please install Python $PYTHON_MIN_VERSION or higher from https://www.python.org/"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_info "Found Python version: $PYTHON_VERSION"
    
    if version_gt "$PYTHON_MIN_VERSION" "$PYTHON_VERSION"; then
        print_error "Python version $PYTHON_VERSION is too old!"
        echo "Please upgrade to Python $PYTHON_MIN_VERSION or higher"
        exit 1
    fi
    
    # Check for pip
    if ! command_exists pip3; then
        print_warning "pip3 not found. Installing pip..."
        python3 -m ensurepip --default-pip
    fi
    
    # Check available memory
    if [[ "$OS_TYPE" == "Linux" ]]; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $TOTAL_MEM -lt 8 ]]; then
            print_warning "System has less than 8GB RAM. Large datasets may cause issues."
        fi
    fi
    
    print_success "System requirements check passed!"
}

# Function to create virtual environment
create_virtual_environment() {
    print_info "Creating virtual environment..."
    
    # Check if venv already exists
    if [[ -d "$VENV_NAME" ]]; then
        read -p "Virtual environment '$VENV_NAME' already exists. Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
        else
            print_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    # Create virtual environment
    python3 -m venv "$VENV_NAME"
    
    # Activate virtual environment
    if [[ "$OS_TYPE" == "Windows" ]]; then
        source "$VENV_NAME/Scripts/activate"
    else
        source "$VENV_NAME/bin/activate"
    fi
    
    # Upgrade pip in virtual environment
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created successfully!"
}

# Function to generate requirements files
generate_requirements_files() {
    print_info "Generating requirements files..."
    
    # Create requirements-core.txt (core dependencies)
    cat > requirements-core.txt << 'EOF'
# Core Data Processing
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
sklearn-genetic-opt>=0.10.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Utilities
joblib>=1.1.0
pyyaml>=5.4.0
click>=8.0.0
colorlog>=6.6.0
tqdm>=4.62.0

# File Format Support
pyarrow>=7.0.0
openpyxl>=3.0.0
EOF

    # Create requirements-full.txt (all optional dependencies)
    cat > requirements-full.txt << 'EOF'
# Include all core requirements
-r requirements-core.txt

# Deep Learning (optional)
tensorflow>=2.8.0
torch>=1.10.0

# Advanced ML
catboost>=1.0.0
imbalanced-learn>=0.9.0

# Experiment Tracking
mlflow>=1.25.0
optuna>=2.10.0
toml>=0.10.2

# Model Interpretability
shap>=0.40.0
lime>=0.2.0.1

# Data Quality
evidently>=0.1.50
great-expectations>=0.15.0

# Advanced Visualization
plotly>=5.5.0
altair>=4.2.0
colorama>=0.4.6

# Geospatial (optional)
geopandas>=0.9.0

# Additional File Formats
h5py>=3.6.0
xlrd>=2.0.0

# Parallel/Distributed Computing
dask[complete]>=2023.1.0
distributed>=2023.1.0
ray[default]>=2.0.0
ray[tune]>=2.0.0 
EOF

    # Create requirements-dev.txt (development dependencies)
    cat > requirements-dev.txt << 'EOF'
# Include all full requirements
-r requirements-full.txt

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-mock>=3.6.0

# Code Quality
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
isort>=5.10.0

# Documentation
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0

# Development Tools
ipython>=8.0.0
jupyter>=1.0.0
jupyterlab>=3.3.0
EOF

    print_success "Requirements files generated!"
}

# Function to install dependencies
install_dependencies() {
    local install_type=$1
    
    print_info "Installing $install_type dependencies..."
    
    case $install_type in
        "basic")
            pip install -r requirements-core.txt
            ;;
        "full")
            pip install -r requirements-full.txt
            ;;
        "dev")
            pip install -r requirements-dev.txt
            ;;
        *)
            print_error "Unknown installation type: $install_type"
            exit 1
            ;;
    esac
    
    print_success "$install_type dependencies installed successfully!"
}

# Function to create project structure
create_project_structure() {
    print_info "Creating project directory structure..."
    
    # Create necessary directories
    directories=(
        "data/raw"
        "data/processed"
        "data/external"
        "outputs"
        "logs"
        "configs"
        "notebooks"
        "docs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    done
    
    # Create .gitignore if it doesn't exist
    if [[ ! -f .gitignore ]]; then
        print_info "Creating .gitignore file..."
        cat > .gitignore << 'EOF'
# Virtual Environment
data_science_env/
venv/
env/
ENV/

# Data files (customize as needed)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Model files
models/*.pkl
models/*.h5
models/*.pt

# Outputs
outputs/*
!outputs/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment variables
.env
.env.local

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/
EOF
    fi
    
    # Create .gitkeep files to preserve empty directories
    for dir in "${directories[@]}"; do
        touch "$dir/.gitkeep"
    done
    
    print_success "Project structure created!"
}

# Function to create sample configuration
create_sample_config() {
    print_info "Creating sample configuration..."
    
    cat > sample_config.yaml << 'EOF'
# Sample configuration for Data Science Toolkit

experiment:
  name: "sample_experiment"
  description: "Sample configuration demonstrating toolkit usage"
  task_type: "regression"  # or "classification"
  random_state: 42

data:
  train_path: "data/raw/train.csv"
  test_path: "data/raw/test.csv"
  target_column: "target"
  feature_columns: null  # null means use all columns except target
  
preprocessing:
  numeric_features: null  # auto-detect
  categorical_features: null  # auto-detect
  scaling_method: "standard"  # options: standard, minmax, robust
  encoding_method: "onehot"  # options: onehot, label, target
  handle_missing: "impute"  # options: impute, drop
  remove_outliers: false
  outlier_method: "iqr"  # options: iqr, zscore, isolation_forest
  
feature_engineering:
  create_polynomial: false
  polynomial_degree: 2
  create_interactions: false
  feature_selection: true
  selection_method: "mutual_info"  # options: mutual_info, chi2, anova
  n_features: 20
  
training:
  models:
    - name: "random_forest"
      params:
        n_estimators: 100
        max_depth: null
        min_samples_split: 2
        min_samples_leaf: 1
    - name: "xgboost"
      params:
        n_estimators: 100
        learning_rate: 0.1
        max_depth: 6
    - name: "lightgbm"
      params:
        n_estimators: 100
        learning_rate: 0.1
        num_leaves: 31
  
  cross_validation:
    enabled: true
    n_folds: 5
    shuffle: true
    
  hyperparameter_tuning:
    enabled: false
    method: "grid"  # options: grid, random, bayesian
    n_trials: 100
    
evaluation:
  metrics:
    - "rmse"
    - "mae"
    - "r2"
    - "mape"
  create_plots: true
  save_plots: true
  
output:
  save_models: true
  save_predictions: true
  save_feature_importance: true
  save_experiment_info: true
EOF
    
    print_success "Sample configuration created!"
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test basic imports
    python3 -c "
import pandas as pd
import numpy as np
import sklearn
import matplotlib
print('✓ Core packages imported successfully')
" || {
        print_error "Failed to import core packages!"
        return 1
    }
    
    # Check optional packages
    print_info "Checking optional packages..."
    
    optional_packages=("xgboost" "lightgbm" "tensorflow" "torch" "mlflow")
    for package in "${optional_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_success "$package is installed"
        else
            print_warning "$package is not installed (optional)"
        fi
    done
    
    print_success "Installation verified!"
}

# Function to create test script
create_test_script() {
    print_info "Creating test script..."
    
    cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify toolkit installation."""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name}: {e}")
        return False

def main():
    """Run installation tests."""
    print("Testing Data Science Toolkit installation...\n")
    
    # Test core packages
    print("Core packages:")
    core_packages = [
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
        "yaml",
        "click",
        "colorlog",
        "tqdm"
    ]
    
    core_success = all(test_import(pkg) for pkg in core_packages)
    
    print("\nOptional packages:")
    optional_packages = [
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("catboost", "CatBoost"),
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("mlflow", "MLflow"),
        ("optuna", "Optuna"),
        ("shap", "SHAP"),
        ("lime", "LIME"),
        ("evidently", "Evidently")
    ]
    
    for module, name in optional_packages:
        test_import(module, name)
    
    # Test data creation
    print("\nTesting basic functionality:")
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        y = pd.Series(np.random.randn(100), name="target")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        print(f"✓ Basic ML pipeline works (R² score: {score:.4f})")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return 1
    
    if core_success:
        print("\n✅ Installation test completed successfully!")
        return 0
    else:
        print("\n❌ Some core packages failed to import. Please check installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    chmod +x test_installation.py
    print_success "Test script created!"
}

# Main installation function
main() {
    clear
    print_banner
    
    echo "Welcome to the $PROJECT_NAME installation!"
    echo ""
    
    # Check system requirements
    check_system_requirements
    echo ""
    
    # Ask installation type
    print_info "Choose installation type:"
    echo "  1) Basic (core dependencies only)"
    echo "  2) Full (all optional dependencies)"
    echo "  3) Development (full + dev tools)"
    echo "  4) Custom (I'll install manually)"
    echo ""
    
    read -p "Enter your choice (1-4): " choice
    echo ""
    
    # Generate requirements files
    generate_requirements_files
    
    # Create virtual environment
    create_virtual_environment
    
    # Install dependencies based on choice
    case $choice in
        1)
            install_dependencies "basic"
            ;;
        2)
            install_dependencies "full"
            ;;
        3)
            install_dependencies "dev"
            ;;
        4)
            print_info "Skipping automatic installation."
            echo "You can manually install dependencies using:"
            echo "  pip install -r requirements-core.txt"
            ;;
        *)
            print_error "Invalid choice!"
            exit 1
            ;;
    esac
    
    echo ""
    
    # Create project structure
    read -p "Create project directory structure? (Y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        create_project_structure
    fi
    
    echo ""
    
    # Create configuration files
    read -p "Create sample configuration files? (Y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        create_sample_config
    fi
    
    echo ""
    
    # Create test script
    create_test_script
    
    echo ""
    
    # Verify installation
    verify_installation
    
    echo ""
    print_success "Installation completed!"
    echo ""
    
    # Print next steps
    print_info "Next steps:"
    echo "  1. Activate the virtual environment:"
    if [[ "$OS_TYPE" == "Windows" ]]; then
        echo "     source $VENV_NAME/Scripts/activate"
    else
        echo "     source $VENV_NAME/bin/activate"
    fi
    echo "  2. Run the test script:"
    echo "     python test_installation.py"
    echo "  3. Copy .env.example to .env and configure:"
    echo "     cp .env.example .env"
    echo "  4. Check the configs/sample_config.yaml for configuration options"
    echo "  5. Start using the toolkit in your Python scripts!"
    echo ""
    echo "For more information, see the documentation in the docs/ directory."
    echo ""
    
    # Ask to run test
    read -p "Would you like to run the installation test now? (Y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        python3 test_installation.py
    fi
}

# Run main function
main