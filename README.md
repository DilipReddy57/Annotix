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
| **Video Tracking**    | Temporal propagation with SAM3 memory bank         |
| **Multi-Modal RAG**   | Visual + text embeddings for intelligent retrieval |
| **LLM Integration**   | Auto-prompt generation using Google Gemini         |
| **Active Learning**   | Smart sample selection for efficient annotation    |

### 🔬 Advanced AI Agents

| Agent                     | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| **Segmentation Agent**    | SAM3-powered mask generation with text/point/box prompts |
| **Classification Agent**  | CLIP-based semantic labeling                             |
| **RAG Agent**             | ChromaDB vector search for label consistency             |
| **Multi-Modal RAG**       | Combined image + text embeddings                         |
| **QA Agent**              | Confidence scoring and validation                        |
| **Active Learning Agent** | Uncertainty-based sample selection                       |
| **Context Learner**       | Domain adaptation and context understanding              |
| **Instance Learner**      | Few-shot learning for custom objects                     |
| **Counting Agent**        | Object counting with density estimation                  |
| **Tracking Agent**        | Video object tracking with ID persistence                |
| **Scene Graph Engine**    | Spatial relationship detection                           |
| **Embedding Visualizer**  | UMAP/t-SNE clustering visualization                      |
| **LLM Agent**             | Gemini-powered auto-prompt generation                    |
| **Live Stream Agent**     | Real-time video annotation                               |

### 🎨 Modern UI/UX

- **Pastel Theme**: Soft, modern color palette with glassmorphism effects
- **Smart Intro Animation**: Personalized onboarding experience
- **Bento Grid Dashboard**: Asymmetric layout for visual hierarchy
- **Dark Mode**: Full dark theme with emerald accents
- **Smooth Animations**: GPU-accelerated CSS transitions
- **Responsive Design**: Works on desktop and tablet

### 📊 Enterprise Features

| Feature                 | Description                                      |
| ----------------------- | ------------------------------------------------ |
| **Dataset Import**      | Import from Kaggle, HuggingFace, GitHub, or URLs |
| **COCO Export**         | Standard format export for ML pipelines          |
| **Real-time Dashboard** | Project stats, activity feed, system status      |
| **User Authentication** | JWT-based auth with role management              |
| **Feedback System**     | User feedback collection for improvement         |
| **Settings Panel**      | Comprehensive configuration options              |

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
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 2. Start the Frontend

```bash
cd frontend
npm run dev
# UI available at http://localhost:5173
```

### 3. Using the Web Interface

1. **Login**: Use default credentials or register a new account
2. **Create Project**: Click "New Project" on the dashboard
3. **Upload Images**: Drag & drop images or import from Kaggle/URL
4. **Annotate**: Use text prompts for automatic annotation
5. **Export**: Download annotations in COCO format

### 4. Python API Usage

```python
from backend.pipeline.orchestrator import AnnotationPipeline

# Initialize pipeline
pipeline = AnnotationPipeline()

# Basic annotation with text prompt
result = pipeline.process_image(
    "image.jpg",
    prompt="cars and pedestrians"
)

# Video tracking
result = pipeline.process_video(
    "video.mp4",
    prompt="person"
)

# Export to COCO format
pipeline.export_coco("output_annotations.json")
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ANNOTIX                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Frontend  │  │   FastAPI   │  │    CLI      │   Interfaces │
│  │ (React 19)  │  │   Backend   │  │   Client    │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                     │
│  ───────┴────────────────┴────────────────┴───────────────────  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    API ROUTES                             │   │
│  │  /projects  /auth  /export  /counting  /live  /qa  /rag  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 ANNOTATION PIPELINE                       │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │  SAM3   │→ │   RAG   │→ │   QA    │→ │ Export  │      │   │
│  │  │ Agent   │  │  Agent  │  │  Agent  │  │         │      │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 ADVANCED AI AGENTS                        │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │ Active  │  │ Context │  │Instance │  │ Scene   │      │   │
│  │  │Learning │  │ Learner │  │Learner  │  │ Graph   │      │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘      │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │   │
│  │  │Counting │  │Tracking │  │  LLM    │  │  Live   │      │   │
│  │  │ Agent   │  │  Agent  │  │  Agent  │  │ Stream  │      │   │
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
Annotix/
├── backend/
│   ├── agents/                 # AI agents
│   │   ├── segmentation.py     # SAM3 segmentation agent
│   │   ├── rag.py              # RAG agent with ChromaDB
│   │   ├── multimodal_rag.py   # Multi-modal RAG
│   │   ├── qa.py               # Quality assurance
│   │   ├── active_learning.py  # Sample selection
│   │   ├── context_learner.py  # Domain adaptation
│   │   ├── instance_learner.py # Few-shot learning
│   │   ├── counting_agent.py   # Object counting
│   │   ├── tracking_agent.py   # Video tracking
│   │   ├── live_stream.py      # Real-time processing
│   │   ├── llm_agent.py        # Gemini integration
│   │   ├── graph_engine.py     # Scene graphs
│   │   ├── embedding_visualizer.py  # UMAP/t-SNE
│   │   └── aggregator.py       # Result aggregation
│   ├── api/
│   │   └── routes/
│   │       ├── projects.py     # Project management
│   │       ├── auth.py         # Authentication
│   │       ├── export.py       # COCO export
│   │       ├── counting.py     # Counting endpoints
│   │       ├── live.py         # Live stream endpoints
│   │       ├── qa.py           # QA endpoints
│   │       ├── rag.py          # RAG endpoints
│   │       ├── feedback.py     # User feedback
│   │       ├── tasks.py        # Background tasks
│   │       └── system.py       # System status
│   ├── core/
│   │   ├── config.py           # Configuration
│   │   ├── database.py         # SQLite setup
│   │   ├── models.py           # SQLAlchemy models
│   │   └── security.py         # Auth utilities
│   ├── pipeline/
│   │   └── orchestrator.py     # Main pipeline
│   ├── sam3/                   # SAM3 model (submodule)
│   ├── utils/                  # Utilities
│   ├── cli.py                  # CLI interface
│   └── main.py                 # FastAPI entry point
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Home.tsx        # Home page
│   │   │   ├── Layout.tsx      # App layout
│   │   │   ├── Gallery.tsx     # Image gallery
│   │   │   ├── Analytics.tsx   # Analytics dashboard
│   │   │   ├── Settings.tsx    # Settings panel
│   │   │   ├── IntroScreen.tsx # Animated intro
│   │   │   ├── KnowledgeBase.tsx # RAG knowledge base
│   │   │   ├── UploadZone.tsx  # File upload
│   │   │   ├── dashboard/      # Dashboard components
│   │   │   ├── ui/             # UI primitives
│   │   │   ├── Editor/         # Annotation editor
│   │   │   ├── Project/        # Project views
│   │   │   └── Video/          # Video annotation
│   │   ├── api/                # API client
│   │   ├── context/            # React context
│   │   ├── index.css           # Pastel theme styles
│   │   ├── App.tsx             # Main app
│   │   └── main.tsx            # Entry point
│   └── package.json
├── docs/                       # Documentation
├── storage/                    # User uploads (gitignored)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── fly.toml                    # Fly.io deployment
├── railway.toml                # Railway deployment
├── render.yaml                 # Render deployment
├── vercel.json                 # Vercel deployment
└── README.md
```

---

## 🔌 API Endpoints

### Projects

| Method | Endpoint                           | Description            |
| ------ | ---------------------------------- | ---------------------- |
| POST   | `/api/projects/`                   | Create new project     |
| GET    | `/api/projects/`                   | List all projects      |
| GET    | `/api/projects/{id}`               | Get project details    |
| GET    | `/api/projects/stats`              | Dashboard statistics   |
| POST   | `/api/projects/import-dataset`     | Import from Kaggle/URL |
| POST   | `/api/projects/{id}/upload`        | Upload images          |
| POST   | `/api/projects/{id}/videos/upload` | Upload videos          |
| GET    | `/api/projects/{id}/images`        | List project images    |

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

### Counting

| Method | Endpoint                          | Description            |
| ------ | --------------------------------- | ---------------------- |
| POST   | `/api/counting/count`             | Count objects in image |
| GET    | `/api/counting/supported-classes` | Get supported classes  |

### Live Stream

| Method | Endpoint           | Description           |
| ------ | ------------------ | --------------------- |
| POST   | `/api/live/start`  | Start live processing |
| POST   | `/api/live/stop`   | Stop live processing  |
| GET    | `/api/live/status` | Get stream status     |

### RAG & Knowledge Base

| Method | Endpoint         | Description           |
| ------ | ---------------- | --------------------- |
| POST   | `/api/rag/query` | Query knowledge base  |
| POST   | `/api/rag/add`   | Add to knowledge base |

### System

| Method | Endpoint             | Description          |
| ------ | -------------------- | -------------------- |
| GET    | `/api/system/status` | System health status |

---

## 🎨 UI Theme

ANNOTIX uses a **Pastel** theme with emerald accents:

| Element     | Color       | Hex       |
| ----------- | ----------- | --------- |
| Primary     | Emerald     | `#10b981` |
| Background  | Dark Carbon | `#09090b` |
| Card        | Charcoal    | `#0f0f12` |
| Success     | Green       | `#22c55e` |
| Warning     | Amber       | `#f59e0b` |
| Error       | Red         | `#ef4444` |
| Pastel Rose | Soft Pink   | `#fecdd3` |
| Pastel Blue | Soft Blue   | `#bfdbfe` |
| Pastel Mint | Soft Green  | `#bbf7d0` |

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

# Frontend tests
cd frontend && npm test
```

### Build for Production

```bash
# Frontend
cd frontend
npm run build

# The dist/ folder can be served statically
```

### Deployment Options

- **Fly.io**: `fly deploy`
- **Railway**: Push to connected repo
- **Render**: Push to connected repo
- **Vercel**: Push to connected repo (frontend only)

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

**Last Updated**: December 2024

</div>
