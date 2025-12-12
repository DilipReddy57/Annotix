# Contributing to ANNOTIX

Thank you for your interest in contributing to ANNOTIX! This document provides guidelines for contributing.

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/DilipReddy57/Annotix.git
cd Annotix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install PyTorch with CUDA (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install development dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for SAM3)
huggingface-cli login

# Frontend setup
cd frontend
npm install
```

### Running the Application

```bash
# Terminal 1: Start Backend
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Frontend
cd frontend
npm run dev
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
- **TypeScript/React**: ESLint with Prettier
- **Type Hints**: All functions should have type annotations
- **Docstrings**: Google-style docstrings for all public functions

### Formatting

```bash
# Format Python code
black backend/
ruff check backend/ --fix

# Sort imports
isort backend/

# Format frontend code
cd frontend
npm run lint
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
- Browser version (for frontend issues)
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
Annotix/
├── backend/
│   ├── agents/         # AI agents (SAM3, RAG, Counting, etc.)
│   ├── api/            # FastAPI routes
│   │   └── routes/     # Individual route modules
│   ├── core/           # Config, database, models
│   ├── pipeline/       # Annotation orchestration
│   ├── sam3/           # SAM3 model (submodule)
│   └── utils/          # Utilities
├── frontend/
│   └── src/
│       ├── components/ # React components
│       ├── api/        # API client
│       └── context/    # React context
└── docs/               # Documentation
```

## 🤖 Adding New Agents

To add a new AI agent:

1. Create a new file in `backend/agents/`
2. Implement the agent class following existing patterns
3. Register the agent in `backend/pipeline/orchestrator.py`
4. Add API routes in `backend/api/routes/` if needed
5. Update tests

Example agent structure:

```python
"""
My New Agent - Description of what it does.
"""
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class MyNewAgent:
    """Agent for doing something awesome."""

    def __init__(self):
        """Initialize the agent."""
        self.initialized = False

    def initialize(self) -> None:
        """Lazy initialization of resources."""
        if self.initialized:
            return
        # Initialize models, etc.
        self.initialized = True

    def process(self, data: Any) -> Dict[str, Any]:
        """Process input and return results."""
        self.initialize()
        # Implementation
        return {"result": "success"}
```

## ✅ Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No sensitive data exposed
- [ ] Changes are backwards compatible
- [ ] New agents follow existing patterns
- [ ] API changes are documented

## 🔐 Security

- Never commit API keys or secrets
- Use environment variables for configuration
- Report security vulnerabilities privately

---

Thank you for contributing! 🎉
