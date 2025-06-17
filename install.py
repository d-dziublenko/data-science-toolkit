#!/usr/bin/env python3
"""
Interactive installation script for Data Science Toolkit
Allows selective installation of packages based on user needs
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

class PackageInstaller:
    """
    A smart package installer that handles dependencies and provides
    an interactive interface for selecting which packages to install.
    """
    
    def __init__(self):
        """Initialize the installer with package definitions and categories."""
        self.installed_packages = set()
        self.failed_packages = set()
        self.package_groups = self._define_package_groups()
        
    def _define_package_groups(self) -> Dict[str, Dict[str, any]]:
        """
        Define all available packages organized by category.
        Each package has metadata about its purpose and dependencies.
        """
        return {
            "build_essentials": {
                "name": "Build Essentials",
                "description": "Core packages required for building other packages",
                "packages": {
                    "pip": {"version": ">=22.0", "description": "Package installer"},
                    "setuptools": {"version": ">=60.0", "description": "Package setup tools"},
                    "wheel": {"version": ">=0.37", "description": "Built package format"},
                    "Cython": {"version": ">=0.29", "description": "C-Extensions for Python"},
                    "numpy": {"version": ">=1.21.0", "description": "Numerical computing", "priority": 1},
                },
                "default": True,
                "required": True  # These are always installed
            },
            "core_scientific": {
                "name": "Core Scientific Libraries",
                "description": "Essential libraries for data science",
                "packages": {
                    "pandas": {"version": ">=1.3.0", "description": "Data manipulation and analysis"},
                    "scipy": {"version": ">=1.7.0", "description": "Scientific computing"},
                    "scikit-learn": {"version": ">=1.0.0", "description": "Machine learning library"},
                },
                "default": True
            },
            "visualization": {
                "name": "Visualization Tools",
                "description": "Libraries for creating plots and visualizations",
                "packages": {
                    "matplotlib": {"version": ">=3.4.0", "description": "Basic plotting library"},
                    "seaborn": {"version": ">=0.11.0", "description": "Statistical data visualization"},
                    "plotly": {"version": ">=5.0.0", "description": "Interactive visualizations"},
                    "yellowbrick": {"version": ">=1.3.0", "description": "ML visualization"},
                },
                "default": True
            },
            "machine_learning": {
                "name": "Machine Learning Libraries",
                "description": "Advanced ML algorithms and tools",
                "packages": {
                    "xgboost": {"version": ">=1.5.0", "description": "Gradient boosting framework"},
                    "lightgbm": {"version": ">=3.0.0", "description": "Fast gradient boosting"},
                    "catboost": {"version": ">=1.0.0", "description": "Gradient boosting with categorical features"},
                    "optuna": {"version": ">=2.10.0", "description": "Hyperparameter optimization"},
                },
                "default": True
            },
            "deep_learning": {
                "name": "Deep Learning Frameworks",
                "description": "Neural network and deep learning libraries",
                "packages": {
                    "torch": {"version": ">=1.10.0", "description": "PyTorch deep learning framework"},
                    "tensorflow": {"version": ">=2.7.0", "description": "TensorFlow deep learning framework"},
                },
                "default": False,
                "note": "Choose either PyTorch or TensorFlow, not both"
            },
            "feature_engineering": {
                "name": "Feature Engineering Tools",
                "description": "Tools for feature creation and selection",
                "packages": {
                    "category_encoders": {"version": ">=2.3.0", "description": "Categorical encoding strategies"},
                    "imbalanced-learn": {"version": ">=0.8.0", "description": "Handle imbalanced datasets"},
                    "featuretools": {"version": ">=1.0.0", "description": "Automated feature engineering"},
                },
                "default": True
            },
            "model_interpretation": {
                "name": "Model Interpretation",
                "description": "Tools for explaining model predictions",
                "packages": {
                    "shap": {"version": ">=0.40.0", "description": "SHAP values for model explanation"},
                    "lime": {"version": ">=0.2.0", "description": "Local model interpretation"},
                },
                "default": True
            },
            "cli_and_logging": {
                "name": "CLI and Logging Tools",
                "description": "Command-line interface and logging utilities",
                "packages": {
                    "click": {"version": ">=8.0.0", "description": "CLI creation framework"},
                    "colorama": {"version": ">=0.4.4", "description": "Cross-platform colored output"},
                    "colorlog": {"version": ">=6.6.0", "description": "Colored logging"},
                    "tqdm": {"version": ">=4.62.0", "description": "Progress bars"},
                    "rich": {"version": ">=10.0.0", "description": "Rich terminal formatting"},
                },
                "default": True
            },
            "file_formats": {
                "name": "File Format Support",
                "description": "Support for various data file formats",
                "packages": {
                    "pyarrow": {"version": ">=6.0.0", "description": "Parquet file support"},
                    "openpyxl": {"version": ">=3.0.0", "description": "Excel file support"},
                    "xlrd": {"version": ">=2.0.0", "description": "Legacy Excel support"},
                    "h5py": {"version": ">=3.0.0", "description": "HDF5 file support"},
                },
                "default": True
            },
            "parallel_computing": {
                "name": "Parallel Computing",
                "description": "Tools for parallel and distributed processing",
                "packages": {
                    "joblib": {"version": ">=1.1.0", "description": "Parallel computing tools"},
                    "dask[complete]": {"version": ">=2021.10.0", "description": "Parallel computing framework"},
                    "ray": {"version": ">=1.9.0", "description": "Distributed computing"},
                },
                "default": False
            },
            "statistics": {
                "name": "Statistical Analysis",
                "description": "Advanced statistical modeling tools",
                "packages": {
                    "statsmodels": {"version": ">=0.12.0", "description": "Statistical modeling"},
                    "pmdarima": {"version": ">=1.8.0", "description": "Auto-ARIMA models"},
                    "prophet": {"version": ">=1.0.0", "description": "Time series forecasting"},
                },
                "default": False
            },
            "web_and_api": {
                "name": "Web and API Tools",
                "description": "Tools for web requests and API development",
                "packages": {
                    "requests": {"version": ">=2.26.0", "description": "HTTP library"},
                    "aiohttp": {"version": ">=3.8.0", "description": "Async HTTP client/server"},
                    "fastapi": {"version": ">=0.70.0", "description": "Modern web API framework"},
                    "streamlit": {"version": ">=1.0.0", "description": "Web app framework for ML"},
                },
                "default": False
            },
            "database_drivers": {
                "name": "Database Drivers",
                "description": "Connectors for various databases (may require system dependencies)",
                "packages": {
                    "psycopg2-binary": {"version": ">=2.9.0", "description": "PostgreSQL driver"},
                    "pymongo": {"version": ">=3.12.0", "description": "MongoDB driver"},
                    "sqlalchemy": {"version": ">=1.4.0", "description": "SQL toolkit and ORM"},
                    "pymysql": {"version": ">=1.0.0", "description": "Pure Python MySQL driver"},
                },
                "default": False,
                "note": "Database drivers may require additional system dependencies"
            },
            "development_tools": {
                "name": "Development Tools",
                "description": "Tools for testing and development",
                "packages": {
                    "pytest": {"version": ">=6.2.0", "description": "Testing framework"},
                    "pytest-cov": {"version": ">=3.0.0", "description": "Test coverage"},
                    "black": {"version": ">=21.0", "description": "Code formatter"},
                    "flake8": {"version": ">=4.0.0", "description": "Code linter"},
                    "mypy": {"version": ">=0.910", "description": "Type checker"},
                },
                "default": False
            },
            "jupyter": {
                "name": "Jupyter Ecosystem",
                "description": "Jupyter notebooks and extensions",
                "packages": {
                    "jupyter": {"version": ">=1.0.0", "description": "Jupyter notebook"},
                    "jupyterlab": {"version": ">=3.1.0", "description": "JupyterLab interface"},
                    "ipywidgets": {"version": ">=7.6.0", "description": "Interactive widgets"},
                    "nbconvert": {"version": ">=6.0.0", "description": "Notebook conversion"},
                },
                "default": False
            },
            "utilities": {
                "name": "Utility Packages",
                "description": "General purpose utilities",
                "packages": {
                    "pyyaml": {"version": ">=5.4.0", "description": "YAML parser"},
                    "python-dotenv": {"version": ">=0.19.0", "description": "Environment variable management"},
                    "humanize": {"version": ">=3.12.0", "description": "Human-readable numbers"},
                    "tabulate": {"version": ">=0.8.0", "description": "Table formatting"},
                },
                "default": True
            }
        }
    
    def run_command(self, cmd: str, description: str) -> bool:
        """
        Execute a shell command and handle the output gracefully.
        
        This method runs the command and provides clear feedback about
        success or failure, making debugging easier for users.
        """
        print(f"\n{'─'*60}")
        print(f"🔧 {description}")
        print(f"{'─'*60}")
        print(f"Command: {cmd}")
        
        try:
            # Run the command with real-time output
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"    {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print("✅ Success!")
                return True
            else:
                print(f"❌ Failed with exit code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
    
    def check_python_version(self) -> bool:
        """
        Verify that the Python version meets the minimum requirements.
        
        The toolkit requires Python 3.8 or higher for full compatibility
        with all features, especially type hints and newer syntax.
        """
        version = sys.version_info
        print(f"\n📌 Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required!")
            print("   Please upgrade your Python installation.")
            return False
        
        print("✅ Python version is compatible")
        return True
    
    def display_package_group(self, group_id: str, group_data: Dict) -> None:
        """Display information about a package group in a formatted way."""
        print(f"\n📦 {group_data['name']}")
        print(f"   {group_data['description']}")
        
        if 'note' in group_data:
            print(f"   ⚠️  Note: {group_data['note']}")
        
        print("\n   Packages:")
        for pkg_name, pkg_info in group_data['packages'].items():
            print(f"   • {pkg_name:<25} - {pkg_info['description']}")
    
    def get_user_selections(self) -> Dict[str, List[str]]:
        """
        Interactive menu for users to select which packages to install.
        
        This method presents users with a clear menu of all available
        package groups and lets them choose exactly what they need.
        """
        selections = {}
        
        print("\n" + "="*70)
        print("📋 PACKAGE SELECTION")
        print("="*70)
        print("\nThis installer will help you choose exactly which packages to install.")
        print("You can select entire groups or individual packages within each group.")
        
        # First, let the user choose quick installation options
        print("\n🚀 Quick Installation Options:")
        print("1. Minimal (Core only)")
        print("2. Recommended (Default selections)")
        print("3. Full (Everything except conflicting packages)")
        print("4. Custom (Choose group by group)")
        
        quick_choice = input("\nSelect an option (1-4) [default: 2]: ").strip() or "2"
        
        if quick_choice == "1":
            # Minimal installation
            return self._get_minimal_selection()
        elif quick_choice == "2":
            # Recommended installation
            return self._get_recommended_selection()
        elif quick_choice == "3":
            # Full installation
            return self._get_full_selection()
        else:
            # Custom installation
            return self._get_custom_selection()
    
    def _get_minimal_selection(self) -> Dict[str, List[str]]:
        """Get minimal package selection (core scientific stack only)."""
        return {
            "build_essentials": list(self.package_groups["build_essentials"]["packages"].keys()),
            "core_scientific": list(self.package_groups["core_scientific"]["packages"].keys()),
            "visualization": ["matplotlib", "seaborn"],
            "cli_and_logging": ["click", "colorama", "colorlog", "tqdm"],
            "utilities": ["pyyaml"]
        }
    
    def _get_recommended_selection(self) -> Dict[str, List[str]]:
        """Get recommended package selection based on default flags."""
        selections = {}
        for group_id, group_data in self.package_groups.items():
            if group_data.get('default', False) or group_data.get('required', False):
                selections[group_id] = list(group_data['packages'].keys())
        return selections
    
    def _get_full_selection(self) -> Dict[str, List[str]]:
        """Get full package selection (everything except conflicting packages)."""
        selections = {}
        for group_id, group_data in self.package_groups.items():
            if group_id == "deep_learning":
                # For deep learning, ask which framework to use
                print("\n🤖 Which deep learning framework do you prefer?")
                print("1. PyTorch")
                print("2. TensorFlow")
                print("3. Both (not recommended)")
                print("4. None")
                
                dl_choice = input("Select (1-4) [default: 1]: ").strip() or "1"
                if dl_choice == "1":
                    selections[group_id] = ["torch"]
                elif dl_choice == "2":
                    selections[group_id] = ["tensorflow"]
                elif dl_choice == "3":
                    selections[group_id] = ["torch", "tensorflow"]
                # else: skip deep learning
            else:
                selections[group_id] = list(group_data['packages'].keys())
        
        return selections
    
    def _get_custom_selection(self) -> Dict[str, List[str]]:
        """Get custom package selection with detailed user interaction."""
        selections = {}
        
        for group_id, group_data in self.package_groups.items():
            # Skip required groups (they're always installed)
            if group_data.get('required', False):
                selections[group_id] = list(group_data['packages'].keys())
                continue
            
            self.display_package_group(group_id, group_data)
            
            print(f"\n   Options for {group_data['name']}:")
            print("   a) Install all packages in this group")
            print("   s) Select individual packages")
            print("   n) Skip this group")
            
            default = 'a' if group_data.get('default', False) else 'n'
            choice = input(f"   Choose (a/s/n) [default: {default}]: ").strip().lower() or default
            
            if choice == 'a':
                selections[group_id] = list(group_data['packages'].keys())
                print(f"   ✅ Selected all packages in {group_data['name']}")
            elif choice == 's':
                # Individual package selection
                selected_packages = []
                print("\n   Select individual packages (y/n for each):")
                
                for pkg_name, pkg_info in group_data['packages'].items():
                    pkg_choice = input(f"   Install {pkg_name} ({pkg_info['description']})? (y/n) [y]: ").strip().lower() or 'y'
                    if pkg_choice == 'y':
                        selected_packages.append(pkg_name)
                
                if selected_packages:
                    selections[group_id] = selected_packages
                    print(f"   ✅ Selected {len(selected_packages)} packages from {group_data['name']}")
            else:
                print(f"   ⏭️  Skipping {group_data['name']}")
        
        return selections
    
    def install_packages(self, selections: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
        """
        Install the selected packages in the correct order.
        
        This method handles the actual installation process, ensuring that
        dependencies are installed first and handling any failures gracefully.
        """
        print("\n" + "="*70)
        print("📦 STARTING INSTALLATION")
        print("="*70)
        
        total_packages = sum(len(packages) for packages in selections.values())
        print(f"\nTotal packages to install: {total_packages}")
        
        # First, always upgrade pip, setuptools, and wheel
        print("\n🔄 Updating package management tools...")
        self.run_command(
            f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
            "Upgrading pip, setuptools, and wheel"
        )
        
        # Install packages group by group
        installed_count = 0
        failed_packages = []
        
        # Define installation order (some groups should be installed before others)
        install_order = [
            "build_essentials",
            "core_scientific",
            "visualization",
            "feature_engineering",
            "machine_learning",
            "model_interpretation",
            "cli_and_logging",
            "file_formats",
            "utilities",
            "statistics",
            "parallel_computing",
            "deep_learning",
            "web_and_api",
            "database_drivers",
            "jupyter",
            "development_tools"
        ]
        
        for group_id in install_order:
            if group_id not in selections:
                continue
            
            group_data = self.package_groups[group_id]
            packages = selections[group_id]
            
            if not packages:
                continue
            
            print(f"\n📂 Installing {group_data['name']}...")
            print(f"{'─'*60}")
            
            for package in packages:
                if package not in group_data['packages']:
                    continue
                
                pkg_info = group_data['packages'][package]
                pkg_spec = f"{package}{pkg_info['version']}"
                
                # Special handling for certain packages
                if package == "torch" and self._cuda_available():
                    # Install PyTorch with CUDA support
                    pkg_spec = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
                
                success = self.run_command(
                    f"{sys.executable} -m pip install {pkg_spec}",
                    f"Installing {package} - {pkg_info['description']}"
                )
                
                if success:
                    installed_count += 1
                    self.installed_packages.add(package)
                else:
                    failed_packages.append((package, group_data['name'], pkg_info['description']))
                    self.failed_packages.add(package)
        
        # Summary
        print("\n" + "="*70)
        print("📊 INSTALLATION SUMMARY")
        print("="*70)
        print(f"\n✅ Successfully installed: {installed_count}/{total_packages} packages")
        
        if failed_packages:
            print(f"\n❌ Failed to install {len(failed_packages)} packages:")
            for pkg, group, desc in failed_packages:
                print(f"   • {pkg} ({group}) - {desc}")
        
        return installed_count == total_packages, failed_packages
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available on the system."""
        return (
            os.path.exists("/usr/local/cuda") or 
            os.path.exists("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA") or
            os.environ.get("CUDA_HOME") is not None
        )
    
    def create_test_script(self, selections: Dict[str, List[str]]) -> None:
        """
        Generate a test script that verifies the installation.
        
        This script helps users confirm that everything was installed
        correctly and provides diagnostic information if something is wrong.
        """
        # Create the test script content
        test_script_content = [
            '#!/usr/bin/env python3',
            '"""',
            'Test script to verify Data Science Toolkit installation.',
            'Generated by the installation script.',
            '"""',
            '',
            'import sys',
            'import importlib',
            'import json',
            '',
            'def test_import(module_name, friendly_name=None):',
            '    """Test if a module can be imported and get its version."""',
            '    if friendly_name is None:',
            '        friendly_name = module_name',
            '    ',
            '    try:',
            '        module = importlib.import_module(module_name)',
            '        version = getattr(module, "__version__", "unknown")',
            '        print(f"✅ {friendly_name:<30} {version}")',
            '        return True, version',
            '    except ImportError as e:',
            '        print(f"❌ {friendly_name:<30} Not installed ({str(e).split(\' \')[0]})")',
            '        return False, None',
            '',
            'def test_functionality():',
            '    """Test basic functionality of key packages."""',
            '    print("\\n" + "="*60)',
            '    print("🧪 FUNCTIONALITY TESTS")',
            '    print("="*60)',
            '    ',
            '    tests_passed = 0',
            '    tests_total = 0',
            '    ',
            '    # Test 1: NumPy and Pandas',
            '    tests_total += 1',
            '    try:',
            '        import numpy as np',
            '        import pandas as pd',
            '        ',
            '        data = pd.DataFrame({',
            '            "x": np.random.randn(10),',
            '            "y": np.random.randn(10)',
            '        })',
            '        print("✅ Test 1: Created DataFrame with NumPy and Pandas")',
            '        tests_passed += 1',
            '    except Exception as e:',
            '        print(f"❌ Test 1: NumPy/Pandas test failed - {e}")',
            '    ',
            '    # Test 2: Scikit-learn',
            '    tests_total += 1',
            '    try:',
            '        from sklearn.linear_model import LinearRegression',
            '        from sklearn.model_selection import train_test_split',
            '        ',
            '        X_train, X_test, y_train, y_test = train_test_split(',
            '            data[["x"]], data["y"], test_size=0.3, random_state=42',
            '        )',
            '        model = LinearRegression()',
            '        model.fit(X_train, y_train)',
            '        score = model.score(X_test, y_test)',
            '        print(f"✅ Test 2: Trained LinearRegression model (R² = {score:.3f})")',
            '        tests_passed += 1',
            '    except Exception as e:',
            '        print(f"❌ Test 2: Scikit-learn test failed - {e}")',
            '    ',
            '    # Test 3: Visualization',
            '    tests_total += 1',
            '    try:',
            '        import matplotlib.pyplot as plt',
            '        fig, ax = plt.subplots(figsize=(6, 4))',
            '        ax.plot([1, 2, 3], [1, 4, 9])',
            '        plt.close()',
            '        print("✅ Test 3: Created matplotlib figure")',
            '        tests_passed += 1',
            '    except Exception as e:',
            '        print(f"❌ Test 3: Matplotlib test failed - {e}")',
            '    ',
            '    print(f"\\nFunctionality tests passed: {tests_passed}/{tests_total}")',
            '    return tests_passed == tests_total',
            '',
            'def main():',
            '    """Main test function."""',
            '    print("\\n🔍 DATA SCIENCE TOOLKIT INSTALLATION TEST")',
            '    print("="*60)',
            '    ',
            '    # Load the installation selections',
            '    selections = ' + repr(selections),
            '    ',
            '    # Test each installed package',
            '    all_success = True',
            '    ',
            '    for group_id, packages in selections.items():',
            '        if not packages:',
            '            continue',
            '            ',
            '        print(f"\\n📦 Testing {group_id.replace(\'_\', \' \').title()}")',
            '        print("-"*60)',
            '        ',
            '        group_success = True',
            '        for package in packages:',
            '            # Handle special package names',
            '            import_name = {',
            '                "scikit-learn": "sklearn",',
            '                "pillow": "PIL",',
            '                "beautifulsoup4": "bs4",',
            '                "pytables": "tables",',
            '                "pyyaml": "yaml",',
            '                "python-dotenv": "dotenv",',
            '                "psycopg2-binary": "psycopg2",',
            '                "pymysql": "pymysql",',
            '                "imbalanced-learn": "imblearn",',
            '                "dask[complete]": "dask",',
            '            }.get(package, package)',
            '            ',
            '            success, version = test_import(import_name, package)',
            '            if not success:',
            '                group_success = False',
            '                all_success = False',
            '        ',
            '        if group_success:',
            '            print(f"✅ All packages in {group_id} are working!")',
            '    ',
            '    # Run functionality tests',
            '    print()',
            '    func_success = test_functionality()',
            '    ',
            '    # Final summary',
            '    print("\\n" + "="*60)',
            '    print("📋 SUMMARY")',
            '    print("="*60)',
            '    ',
            '    if all_success and func_success:',
            '        print("\\n✨ Excellent! All packages are installed and working correctly.")',
            '        print("   The Data Science Toolkit is ready to use!")',
            '        ',
            '        print("\\n📚 Next steps:")',
            '        print("   1. Check out the examples/ directory for usage examples")',
            '        print("   2. Read the documentation for detailed guides")',
            '        print("   3. Try running: python -m data_science_toolkit.examples.quickstart")',
            '    else:',
            '        print("\\n⚠️  Some packages or tests failed.")',
            '        print("   The toolkit may still be usable, but some features might be limited.")',
            '        print("\\n💡 Troubleshooting tips:")',
            '        print("   1. Check the error messages above for specific issues")',
            '        print("   2. Try installing failed packages individually")',
            '        print("   3. Ensure you have all required system dependencies")',
            '        print("   4. Check the INSTALL.md file for platform-specific instructions")',
            '',
            'if __name__ == "__main__":',
            '    main()',
        ]
        
        # Write the test script
        with open("test_installation.py", "w") as f:
            f.write('\n'.join(test_script_content))
        
        print("\n✅ Created test_installation.py")
        print("   Run it with: python test_installation.py")
    
    def save_installation_report(self, selections: Dict[str, List[str]], 
                               failed_packages: List[Tuple[str, str, str]]) -> None:
        """Save a detailed installation report for future reference."""
        report = {
            "installation_date": str(sys.stdout),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "selected_packages": selections,
            "installed_packages": list(self.installed_packages),
            "failed_packages": [{"name": p[0], "group": p[1], "description": p[2]} 
                              for p in failed_packages]
        }
        
        with open("installation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n📄 Saved installation report to installation_report.json")
    
    def run(self) -> None:
        """
        Main entry point for the installer.
        
        This method orchestrates the entire installation process,
        from checking prerequisites to generating the test script.
        """
        print("\n🚀 DATA SCIENCE TOOLKIT INTERACTIVE INSTALLER")
        print("="*70)
        print("\nThis installer will help you set up the Data Science Toolkit")
        print("with exactly the packages you need for your projects.")
        
        # Check Python version
        if not self.check_python_version():
            sys.exit(1)
        
        # Get user selections
        selections = self.get_user_selections()
        
        # Confirm selections
        print("\n" + "="*70)
        print("📋 INSTALLATION PLAN")
        print("="*70)
        print("\nYou have selected the following packages:")
        
        total_packages = 0
        for group_id, packages in selections.items():
            if packages:
                group_name = self.package_groups[group_id]['name']
                print(f"\n{group_name}: {len(packages)} packages")
                total_packages += len(packages)
        
        print(f"\nTotal packages to install: {total_packages}")
        
        confirm = input("\nProceed with installation? (y/n) [y]: ").strip().lower() or 'y'
        if confirm != 'y':
            print("\nInstallation cancelled.")
            return
        
        # Install packages
        success, failed_packages = self.install_packages(selections)
        
        # Create test script
        self.create_test_script(selections)
        
        # Save installation report
        self.save_installation_report(selections, failed_packages)
        
        # Final instructions
        print("\n" + "="*70)
        print("🎉 INSTALLATION COMPLETE!")
        print("="*70)
        
        print("\n📋 Next steps:")
        print("1. Run the test script to verify installation:")
        print("   python test_installation.py")
        print("\n2. If any packages failed, you can try installing them individually:")
        print("   pip install <package_name>")
        print("\n3. Check the installation_report.json for a detailed summary")
        print("\n4. Start using the Data Science Toolkit in your projects!")
        
        if failed_packages:
            print("\n⚠️  Note: Some packages failed to install. The toolkit will still")
            print("   work for most use cases, but some features may be unavailable.")


def main():
    """Entry point for the script."""
    installer = PackageInstaller()
    installer.run()


if __name__ == "__main__":
    main()