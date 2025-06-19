# Contributing to Universal Data Science Toolkit

Thank you for your interest in contributing to the Universal Data Science Toolkit! This document provides guidelines and instructions for contributing to this project. I welcome contributions of all kinds, from bug fixes to new features, documentation improvements, and more.

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. I expect all contributors to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members
- Gracefully accept constructive criticism

## 🚀 Getting Started

### Setting Up Your Development Environment

1. **Fork the repository**

   Click the "Fork" button on the GitHub repository page to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/d-dziublenko/data-science-toolkit.git
   cd data-science-toolkit
   ```

3. **Set up the development environment**

   ```bash
   # Run the installation script with development dependencies
   ./install.sh
   # Choose option 3 (Development) when prompted

   # Or manually install development dependencies
   pip install -r requirements-dev.txt
   ```

4. **Create a new branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

### Development Workflow

1. **Make your changes**

   - Write clean, well-documented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

2. **Run tests**

   ```bash
   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_models.py

   # Run with coverage
   pytest --cov=data_science_toolkit
   ```

3. **Check code quality**

   ```bash
   # Format code
   black .

   # Sort imports
   isort .

   # Check linting
   flake8

   # Type checking
   mypy data_science_toolkit
   ```

4. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add new feature" # Use conventional commits
   ```

5. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

   Go to the original repository on GitHub and create a pull request from your fork.

## 📝 Contribution Guidelines

### Code Style

I follow PEP 8 with some modifications:

- Line length: 100 characters (instead of 79)
- Use type hints for all function signatures
- Use docstrings for all public functions and classes
- Follow the Google docstring format

Example:

```python
def process_data(
    data: pd.DataFrame,
    method: str = "standard",
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Process data using specified method.

    Args:
        data: Input DataFrame to process
        method: Processing method ('standard', 'robust', 'minmax')
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Processed DataFrame

    Raises:
        ValueError: If method is not recognized

    Example:
        >>> df_processed = process_data(df, method='robust')
    """
    # Implementation here
    pass
```

### Commit Messages

I use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:

```
feat: add support for PostgreSQL data loading
fix: handle missing values in feature engineering
docs: update installation instructions
test: add unit tests for ensemble models
```

### Testing

All new features must include tests:

1. **Unit tests**: Test individual functions and methods
2. **Integration tests**: Test component interactions
3. **Performance tests**: For performance-critical code

Example test:

```python
import pytest
import pandas as pd
from data_science_toolkit.core import DataPreprocessor

class TestDataPreprocessor:
    def test_handle_missing_values(self):
        # Arrange
        data = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [None, 2, 3, 4]
        })
        preprocessor = DataPreprocessor()

        # Act
        result = preprocessor.handle_missing_values(
            data,
            strategy='mean'
        )

        # Assert
        assert result['A'].isna().sum() == 0
        assert result['B'].isna().sum() == 0
        assert result['A'].iloc[2] == pytest.approx(2.33, 0.01)
```

### Documentation

Documentation is crucial for this project:

1. **Docstrings**: All public APIs must have comprehensive docstrings
2. **Examples**: Include usage examples in docstrings
3. **Tutorials**: Add Jupyter notebooks for complex features
4. **API Reference**: Update when adding new features

### Performance Considerations

When contributing performance-critical code:

1. Profile your code to identify bottlenecks
2. Consider memory usage for large datasets
3. Add benchmarks for performance-critical functions
4. Use parallel processing where appropriate
5. Document performance characteristics

## 🎯 Areas for Contribution

### High Priority

- **Deep Learning Integration**: Add PyTorch/TensorFlow model wrappers
- **Time Series Models**: Expand time series forecasting capabilities
- **Cloud Integration**: Add support for cloud data sources
- **API Development**: Create REST API for model serving
- **Visualization**: Enhance plotting capabilities

### Good First Issues

Look for issues labeled `good first issue` on GitHub. These are typically:

- Documentation improvements
- Simple bug fixes
- Adding unit tests
- Code cleanup tasks

### Feature Requests

Check the issue tracker for `enhancement` labels or propose new features by:

1. Opening an issue to discuss the feature
2. Getting feedback from maintainers
3. Implementing after approval

## 🔍 Review Process

### What I Look For

1. **Code Quality**

   - Clean, readable code
   - Proper error handling
   - Efficient implementations

2. **Testing**

   - Adequate test coverage
   - Tests pass in CI/CD

3. **Documentation**

   - Clear docstrings
   - Updated README if needed
   - Examples provided

4. **Compatibility**
   - Works with Python 3.8+
   - No breaking changes (unless discussed)
   - Cross-platform compatibility

### Review Timeline

- Initial review: Within 3-5 days
- Feedback incorporation: As needed
- Final approval: After all concerns addressed

## 🛠️ Development Tips

### Running Specific Components

```bash
# Run only core module tests
pytest tests/test_core/

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_models.py::TestAutoML::test_classification
```

### Debugging

```python
# Use the built-in logger
from utils import get_logger

logger = get_logger(__name__)
logger.debug("Debugging information")
logger.info("Processing started")
logger.error("Error occurred", exc_info=True)
```

### Memory Profiling

```python
# For memory-intensive operations
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

## 📞 Getting Help

If you need help:

1. Check existing documentation
2. Search closed issues
3. Ask in discussions
4. Open a new issue with the `question` label

## 🎉 Recognition

Contributors will be:

- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Given credit in relevant documentation

## 📋 Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main
- [ ] No merge conflicts

Thank you for contributing to making data science more accessible! 🙏
