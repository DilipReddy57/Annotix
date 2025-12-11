# ANNOTIX: Autonomous AI Annotation Platform

<div align="center">

![ANNOTIX](https://img.shields.io/badge/ANNOT-IX-10b981?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCIgZmlsbD0ibm9uZSI+PHJlY3QgeD0iNCIgeT0iNCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiByeD0iNCIgc3Ryb2tlPSIjMTBiOTgxIiBzdHJva2Utd2lkdGg9IjIuNSIgc3Ryb2tlLWRhc2hhcnJheT0iOCA0IiBmaWxsPSJub25lIi8+PGNpcmNsZSBjeD0iMjQiIGN5PSIyNCIgcj0iMyIgZmlsbD0iIzEwYjk4MSIvPjwvc3ZnPg==)
[![Python](https://img.shields.io/badge/Python-3.12+-10b981?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ef4444?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-059669?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19+-61DAFB?style=flat-square&logo=react)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

**Enterprise-grade autonomous annotation platform powered by SAM3 (Segment Anything Model 3)**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

**ANNOTIX** is a state-of-the-art data annotation platform that leverages Meta's **SAM3** model for intelligent, autonomous image and video annotation. Unlike traditional annotation tools, ANNOTIX uses advanced AI agents to automate the annotation workflow while maintaining human-level accuracy.

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
- **Dataset Import**: Import datasets directly from Kaggle, HuggingFace, or GitHub

### 🎨 Modern UI/UX

- **Artistic Design**: Handcrafted "Emerald Tech" theme with premium aesthetics
- **SVG Logo**: Custom bounding-box + crosshair logo representing annotation
- **Bento Grid Dashboard**: Asymmetric layout for visual hierarchy
- **Smooth Animations**: GPU-accelerated CSS animations
- **Matrix Rain Effect**: Stylized annotation rain background

### 📊 Enterprise Features

- **Scalable Architecture**: Celery task queues, Redis caching
- **Production Monitoring**: Prometheus metrics, structured logging
- **REST API**: Full FastAPI backend with authentication
- **Real-time Dashboard**: React 19 frontend with analytics

---

## 🛠 Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA 12.1+ (recommended)
- Node.js 18+ (for frontend)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/DilipReddy57/Annotix.git
cd Annotix

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
│                         ANNOTIX                                 │
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
│  │  │ Active  │  │ Context │  │Instance │  │ Scene   │      │   │
│  │  │Learning │  │ Learner │  │Learner  │  │ Graph   │      │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    DATA LAYER                             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                   │   │
│  │  │ SQLite  │  │ChromaDB │  │  File   │                   │   │
│  │  │Database │  │ Vectors │  │ Storage │                   │   │
│  │  └─────────┘  └─────────┘  └─────────┘                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
annotix/
├── backend/
│   ├── agents/           # AI agents (SAM3, RAG, QA, etc.)
│   │   ├── segmentation.py    # SAM3 agent
│   │   ├── rag.py             # RAG agent with ChromaDB
│   │   ├── qa.py              # Quality assurance
│   │   ├── active_learning.py # Sample selection
│   │   └── ...
│   ├── api/              # FastAPI routes
│   │   └── routes/
│   ├── core/             # Config, models, database
│   ├── pipeline/         # Orchestrator
│   └── main.py           # Entry point
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   │   ├── ui/       # UI primitives (annotix-logo, etc.)
│   │   │   ├── dashboard/# Dashboard components
│   │   │   └── ...
│   │   ├── index.css     # Emerald theme
│   │   └── App.tsx       # Main app
│   └── package.json
├── data/                 # Uploads and exports
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

### Projects

| Method | Endpoint                           | Description            |
| ------ | ---------------------------------- | ---------------------- |
| POST   | `/api/projects/`                   | Create new project     |
| GET    | `/api/projects/`                   | List all projects      |
| GET    | `/api/projects/stats`              | Dashboard statistics   |
| POST   | `/api/projects/import-dataset`     | Import from Kaggle/URL |
| POST   | `/api/projects/{id}/upload`        | Upload images          |
| POST   | `/api/projects/{id}/videos/upload` | Upload videos          |

### Annotation

| Method | Endpoint                                      | Description         |
| ------ | --------------------------------------------- | ------------------- |
| POST   | `/api/projects/{id}/images/{img_id}/annotate` | Run annotation      |
| POST   | `/api/projects/{id}/images/{img_id}/segment`  | Interactive segment |
| POST   | `/api/projects/{id}/videos/{vid_id}/annotate` | Video annotation    |

### Export

| Method | Endpoint                        | Description      |
| ------ | ------------------------------- | ---------------- |
| GET    | `/api/export/{project_id}/coco` | Export COCO JSON |

---

## 🎨 UI Theme

ANNOTIX uses the **Emerald Tech** theme:

| Element    | Color    | Hex       |
| ---------- | -------- | --------- |
| Primary    | Emerald  | `#10b981` |
| Accent     | Teal     | `#14b8a6` |
| Background | Carbon   | `#09090b` |
| Card       | Charcoal | `#0f0f12` |
| Success    | Green    | `#22c55e` |
| Warning    | Amber    | `#f59e0b` |
| Error      | Red      | `#ef4444` |

### Typography

- **Display**: Outfit (headings)
- **Sans**: Inter (body)
- **Mono**: JetBrains Mono (code)

---

## 🧪 Development

### Run Tests

```bash
# Backend tests
pytest tests/

# E2E test
python test_e2e.py
```

### Build for Production

```bash
# Frontend
cd frontend
npm run build

# The dist/ folder can be served statically
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Meta AI** - SAM3 (Segment Anything Model 3)
- **Google** - Gemini API for LLM integration
- **Hugging Face** - Model hosting and transformers library
- **ChromaDB** - Vector database for RAG

---

<div align="center">

**Built with ❤️ by [Dilip Reddy](https://github.com/DilipReddy57)**

[![GitHub](https://img.shields.io/badge/GitHub-DilipReddy57-10b981?style=flat-square&logo=github)](https://github.com/DilipReddy57)

</div>
