# Contributing to Cortex-AI

Thank you for your interest in contributing to Cortex-AI! This document provides guidelines for contributing.

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/DilipReddy57/Cortex-Ai.git
cd Cortex-Ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install -e backend/sam3

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test file
pytest tests/test_agents.py
```

## 📝 Code Style

We follow these coding standards:

- **Python**: PEP 8, enforced with `black` and `ruff`
- **Type Hints**: All functions should have type annotations
- **Docstrings**: Google-style docstrings for all public functions

### Formatting

```bash
# Format code
black backend/
ruff check backend/ --fix

# Sort imports
isort backend/
```

## 🔀 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### PR Guidelines

- Keep PRs focused on a single feature/fix
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version
- PyTorch version
- CUDA version (if applicable)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## 💡 Feature Requests

Feature requests are welcome! Please describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## 📂 Project Structure

```
backend/
├── agents/         # AI agents (segmentation, RAG, etc.)
├── api/            # FastAPI routes
├── core/           # Config, database, models
├── pipeline/       # Orchestration
├── sam3/           # SAM3 model
└── utils/          # Utilities
```

## ✅ Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No sensitive data exposed
- [ ] Changes are backwards compatible

---

Thank you for contributing! 🎉
