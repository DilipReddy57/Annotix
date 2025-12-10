# Cortex-AI: Autonomous Annotation Platform with SAM3

<div align="center">

![Cortex-AI](https://img.shields.io/badge/Cortex-AI-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3ZnPg==)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat-square&logo=react)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Enterprise-grade autonomous annotation platform powered by SAM3 (Segment Anything with Concepts)**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

Cortex-AI is a state-of-the-art data annotation platform that leverages Meta's **SAM3 (Segment Anything with Concepts)** model for intelligent, autonomous image and video annotation. Unlike traditional annotation tools, Cortex-AI uses advanced AI agents to automate the annotation workflow while maintaining human-level accuracy.

### What Makes SAM3 Special?

SAM3 introduces **open-vocabulary text prompts** - you can segment ANY concept using natural language:

```python
# Traditional models: Limited to fixed categories
# SAM3: 270,000+ unique concepts!

"person wearing blue jacket"      # ✅ Works!
"red car on the left side"        # ✅ Works!
"golden retriever puppy"          # ✅ Works!
"laptop on wooden desk"           # ✅ Works!
```

---

## ✨ Features

### 🧠 Core AI Capabilities

| Feature               | Description                                        |
| --------------------- | -------------------------------------------------- |
| **SAM3 Segmentation** | Open-vocabulary detection with 270K+ concepts      |
| **Video Tracking**    | Temporal propagation with object ID persistence    |
| **Multi-Modal RAG**   | Visual + text embeddings for intelligent retrieval |
| **LLM Integration**   | Auto-prompt generation using Gemini/GPT            |
| **Active Learning**   | Smart sample selection for efficient annotation    |

### 🔬 Advanced Features

- **Auto-Prompt Generation**: LLM analyzes images and generates optimal SAM3 prompts
- **Label Consistency**: RAG-powered knowledge base prevents labeling inconsistencies
- **Embedding Visualization**: UMAP/t-SNE clustering for annotation analysis
- **Quality Assurance**: Multi-metric confidence fusion and validation
- **Scene Graphs**: Spatial relationship detection between objects
- **COCO Export**: Standard format export for ML pipelines

### 📊 Enterprise Features

- **Scalable Architecture**: Celery task queues, Redis caching
- **Production Monitoring**: Prometheus metrics, structured logging
- **REST API**: Full FastAPI backend with authentication
- **Real-time Dashboard**: React frontend with analytics

---

## 🛠 Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.1+ (recommended)
- Node.js 18+ (for frontend)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/DilipReddy57/Cortex-Ai.git
cd Cortex-Ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (required for SAM3 model)
huggingface-cli login
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🚀 Quick Start

### 1. Start the Backend

```bash
# From project root
python -m backend.main
# API available at http://localhost:8000
```

### 2. Start the Frontend

```bash
cd frontend
npm run dev
# UI available at http://localhost:5173
```

### 3. Python API Usage

```python
from backend.pipeline.orchestrator import AnnotationPipeline

# Initialize pipeline
pipeline = AnnotationPipeline()

# Basic annotation
result = pipeline.process_image(
    "image.jpg",
    prompt="cars and pedestrians"
)

# Smart processing with LLM-powered auto-prompts
result = pipeline.smart_process_image(
    "image.jpg",
    use_auto_prompts=True,
    context="autonomous driving"
)

# Video tracking
result = pipeline.process_video(
    "video.mp4",
    prompt="person"
)

# Get next best images to annotate (Active Learning)
next_batch = pipeline.get_next_annotation_batch(
    candidate_images=["img1.jpg", "img2.jpg", ...],
    batch_size=10
)
```

### 4. CLI Usage

```bash
# Process single image
python -m backend.cli process image.jpg --prompt "cars"

# Process directory
python -m backend.cli process ./images --prompt "people"

# View analytics
python -m backend.cli analytics
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CORTEX-AI                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Frontend  │  │   FastAPI   │  │    CLI      │   Interfaces │
│  │   (React)   │  │   Backend   │  │   Client    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                     │
│  ───────┴────────────────┴────────────────┴───────────────────  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 ANNOTATION PIPELINE                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │  SAM3   │→ │   RAG   │→ │   QA    │→ │ Export  │      │   │
│  │  │ Agent   │  │  Agent  │  │  Agent  │  │ Agent   │      │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 ADVANCED AI AGENTS                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │   LLM   │  │ Active  │  │Embedding│  │MultiMod │      │   │
│  │  │  Agent  │  │Learning │  │   Viz   │  │   RAG   │      │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  ChromaDB   │  │   SQLite    │  │  Celery     │   Storage   │
│  │  (Vectors)  │  │  (Metadata) │  │  (Queue)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Cortex-Ai/
├── backend/
│   ├── agents/                 # AI Agent modules
│   │   ├── segmentation.py     # SAM3 segmentation agent
│   │   ├── rag.py              # RAG for label consistency
│   │   ├── llm_agent.py        # LLM for auto-prompts
│   │   ├── active_learning.py  # Smart sample selection
│   │   ├── multimodal_rag.py   # Visual + text RAG
│   │   ├── embedding_visualizer.py  # UMAP/t-SNE
│   │   ├── tracking_agent.py   # Video tracking
│   │   └── ...
│   ├── api/                    # FastAPI routes
│   ├── core/                   # Config, database, models
│   ├── pipeline/               # Annotation orchestrator
│   ├── sam3/                   # SAM3 model (submodule)
│   └── utils/                  # Utilities
├── frontend/                   # React UI
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── pages/              # Page views
│   │   └── api/                # API client
│   └── ...
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── ...
```

---

## 🔌 API Reference

### Endpoints

| Endpoint          | Method | Description               |
| ----------------- | ------ | ------------------------- |
| `/api/tasks`      | POST   | Create annotation task    |
| `/api/tasks/{id}` | GET    | Get task status           |
| `/api/projects`   | CRUD   | Project management        |
| `/api/qa`         | POST   | Quality assurance checks  |
| `/api/export`     | GET    | Export annotations (COCO) |
| `/api/auth`       | POST   | Authentication            |
| `/health`         | GET    | Health check              |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/tasks" \
  -H "Content-Type: application/json" \
  -d '{"image_path": "image.jpg", "prompt": "cars"}'
```

---

## 🔧 Configuration

Environment variables (`.env`):

```env
# Required for SAM3 model download
HF_TOKEN=your_huggingface_token

# Optional: LLM for auto-prompts
GEMINI_API_KEY=your_gemini_key

# Database
DATABASE_URL=sqlite:///database.db

# Device
DEVICE=cuda  # or cpu
```

---

## 📈 Performance

| Metric                  | Value                      |
| ----------------------- | -------------------------- |
| SAM3 Concepts           | 270,000+                   |
| Human-level Performance | 75-80%                     |
| Annotation Speed        | ~40x faster than manual    |
| GPU Memory              | ~4GB (GTX 1650 compatible) |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Meta AI - SAM3](https://github.com/facebookresearch/sam3) - Segment Anything with Concepts
- [Google - Gemini](https://ai.google.dev/) - LLM for auto-prompts
- [ChromaDB](https://www.trychroma.com/) - Vector database for RAG

---

<div align="center">

**Built with ❤️ by Dilip Reddy**

[⭐ Star this repo](https://github.com/DilipReddy57/Cortex-Ai) • [🐛 Report Bug](https://github.com/DilipReddy57/Cortex-Ai/issues) • [💡 Request Feature](https://github.com/DilipReddy57/Cortex-Ai/issues)

</div>
