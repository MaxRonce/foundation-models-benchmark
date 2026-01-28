# Linting and Formatting

This project uses Black, isort, and Ruff for code quality.

## Run Formatters Locally

```bash
# Install tools
pip install black isort ruff

# Format code with Black
black src/

# Sort imports with isort
isort src/

# Check linting with Ruff
ruff check src/

# Auto-fix Ruff issues
ruff check src/ --fix
```

## Pre-commit Hook

Install pre-commit to automatically format before commits:

```bash
pip install pre-commit
pre-commit install
```

## Configuration Files

- **Black**: Uses default settings (88 chars, py310+)
- **isort**: Compatible with Black
- **Ruff**: Configured in `pyproject.toml`
