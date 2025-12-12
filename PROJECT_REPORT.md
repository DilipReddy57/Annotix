# ANNOTIX: Autonomous Dataset Annotation Agent

### SAM3 + Multi-Agent + RAG Architecture

**Project Report - December 2024**

---

## 1. Purpose of the Project

The purpose of ANNOTIX is to **automatically create high-quality, large-scale annotated datasets** for computer vision tasks such as detection, segmentation, and tracking **without requiring human labeling effort**.

The system uses:

- **SAM3** (Segment Anything Model 3) for state-of-the-art segmentation
- A **multi-agent framework** for label assignment and verification
- **RAG** (Retrieval-Augmented Generation) for maintaining label consistency
- **LLM Integration** (Google Gemini) for intelligent prompt generation

The goal is to build a pipeline that can **convert raw, unlabeled images or videos into COCO-style machine-learning-ready datasets**, enabling teams to train models faster and more accurately.

---

## 2. Why This Project Is Needed

### Manual dataset annotation is one of the biggest bottlenecks in AI/ML development.

Today, computer vision datasets require:

- Thousands to millions of annotated images
- Precise segmentation masks
- Consistent labels across the dataset
- Quality assurance for correctness

**Human annotation is extremely slow, expensive, and inconsistent.**
A skilled annotator can only annotate ~100–200 images/day, and accuracy varies.

For companies building AI models, dataset creation often consumes:

- 40–60 percent of project time
- 50–70 percent of total budget
- Massive workforce for labeling tasks

This slows down model development and limits rapid iterations.

---

## 3. Problem Statement

### "How can we automatically generate accurate, consistent, high-quality segmentation annotations for large-scale image or video datasets without human effort?"

More specifically:

1. Raw datasets contain millions of frames that are unlabeled
2. Manual annotation is slow and error-prone
3. Label inconsistency (e.g., "helmet" vs "headgear") reduces model accuracy
4. Existing segmentation models only give masks but not validated labels
5. There is no pipeline that combines segmentation, classification, QA, and consistency checks autonomously

ANNOTIX solves all these problems with an **intelligent, agent-based automated pipeline**.

---

## 4. Key Features Implemented

### Core AI Pipeline

| Feature           | Description                                     | Status      |
| ----------------- | ----------------------------------------------- | ----------- |
| SAM3 Segmentation | Open-vocabulary detection with 270K+ concepts   | ✅ Complete |
| Text Prompts      | Natural language annotation instructions        | ✅ Complete |
| Video Tracking    | Temporal propagation with object ID persistence | ✅ Complete |
| Multi-Modal RAG   | Visual + text embeddings for consistency        | ✅ Complete |
| Active Learning   | Smart sample selection                          | ✅ Complete |
| COCO Export       | Standard ML format export                       | ✅ Complete |

### AI Agents

| Agent                 | Purpose                           | Status      |
| --------------------- | --------------------------------- | ----------- |
| Segmentation Agent    | SAM3-powered mask generation      | ✅ Complete |
| Classification Agent  | CLIP-based semantic labeling      | ✅ Complete |
| RAG Agent             | Label consistency via ChromaDB    | ✅ Complete |
| Multi-Modal RAG Agent | Combined image + text embeddings  | ✅ Complete |
| QA Agent              | Confidence scoring and validation | ✅ Complete |
| Active Learning Agent | Uncertainty-based selection       | ✅ Complete |
| Context Learner       | Domain adaptation                 | ✅ Complete |
| Instance Learner      | Few-shot object learning          | ✅ Complete |
| Counting Agent        | Object counting with density      | ✅ Complete |
| Tracking Agent        | Video object tracking             | ✅ Complete |
| Scene Graph Engine    | Spatial relationships             | ✅ Complete |
| LLM Agent             | Gemini-powered prompts            | ✅ Complete |
| Live Stream Agent     | Real-time processing              | ✅ Complete |
| Embedding Visualizer  | UMAP/t-SNE clustering             | ✅ Complete |

### Web Application

| Feature             | Description                      | Status      |
| ------------------- | -------------------------------- | ----------- |
| Modern Dashboard    | Bento grid layout with stats     | ✅ Complete |
| Project Management  | Create, import, manage projects  | ✅ Complete |
| Image Gallery       | Grid view with status indicators | ✅ Complete |
| Dataset Import      | Kaggle, HuggingFace, GitHub, URL | ✅ Complete |
| Annotation Editor   | Interactive annotation interface | ✅ Complete |
| Analytics           | Progress charts and metrics      | ✅ Complete |
| Settings            | Comprehensive configuration      | ✅ Complete |
| User Authentication | JWT-based auth                   | ✅ Complete |
| Pastel Theme        | Modern, soft color palette       | ✅ Complete |
| Smart Intro         | Animated onboarding experience   | ✅ Complete |

---

## 5. How the Project Works

### Step-by-Step Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      USER UPLOADS                           │
│                    (Images or Videos)                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   SEGMENTATION AGENT                        │
│              (SAM3 - Mask Generation)                       │
│  • Identifies objects using text prompts                    │
│  • Generates precise segmentation masks                     │
│  • Extracts polygons and bounding boxes                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  CLASSIFICATION AGENT                       │
│               (CLIP - Label Assignment)                     │
│  • Computes image embeddings                                │
│  • Matches against known class embeddings                   │
│  • Assigns semantic labels to masks                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG AGENT                              │
│           (ChromaDB - Consistency Check)                    │
│  • Retrieves similar past annotations                       │
│  • Ensures label naming consistency                         │
│  • Prevents semantic drift                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       QA AGENT                              │
│              (Quality Assurance)                            │
│  • Validates mask precision                                 │
│  • Scores confidence levels                                 │
│  • Flags low-quality annotations                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGGREGATOR AGENT                         │
│              (Result Combination)                           │
│  • Combines all agent outputs                               │
│  • Generates final COCO JSON                                │
│  • Stores in database                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXPORT / DISPLAY                        │
│  • COCO format download                                     │
│  • Frontend visualization                                   │
│  • Analytics dashboard                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Stack

### Backend

| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python 3.12+  | Core language             |
| FastAPI       | REST API framework        |
| PyTorch 2.5+  | Deep learning framework   |
| SAM3          | Segmentation model        |
| CLIP          | Classification embeddings |
| ChromaDB      | Vector database for RAG   |
| SQLite        | Metadata storage          |
| Google Gemini | LLM for prompts           |
| Transformers  | Model loading             |

### Frontend

| Technology   | Purpose      |
| ------------ | ------------ |
| React 19     | UI framework |
| TypeScript   | Type safety  |
| Vite 7       | Build tool   |
| Vanilla CSS  | Styling      |
| React Router | Navigation   |

---

## 7. Impact of the Project

### 1. Reduces annotation cost by 70–90 percent

No human workforce needed for manual labeling.

### 2. Accelerates dataset creation

Millions of images can be annotated in hours instead of weeks.

### 3. Improves model performance

Consistent, high-quality masks → better training → higher accuracy.

### 4. Standardizes output formats

COCO-compatible dataset → works with YOLO, Detectron2, Mask R-CNN, etc.

### 5. Scales easily to enterprise level

Modular agent architecture for easy extension.

---

## 8. Real-World Use Cases

| Industry                | Application                                |
| ----------------------- | ------------------------------------------ |
| **Computer Vision**     | Training detection and segmentation models |
| **Autonomous Vehicles** | Road scene annotation                      |
| **Retail/Security**     | CCTV object detection                      |
| **Robotics**            | Scene understanding for manipulation       |
| **Medical Imaging**     | Organ and tool segmentation                |
| **Media/Broadcasting**  | Video frame annotation                     |

---

## 9. What Makes ANNOTIX Unique

Most annotation tools only use SAM3 to draw masks.

ANNOTIX provides a **full autonomous pipeline** with:

- ✅ Multi-agent orchestration
- ✅ Segmentation with SAM3
- ✅ Classification with CLIP
- ✅ Consistency with RAG
- ✅ Quality assurance
- ✅ Video tracking
- ✅ Active learning
- ✅ Object counting
- ✅ LLM integration
- ✅ Modern web interface
- ✅ Standard COCO export

---

## 10. Future Roadmap

- [ ] Multi-user collaboration with roles
- [ ] Cloud deployment (AWS/GCP)
- [ ] Model fine-tuning interface
- [ ] Real-time collaboration
- [ ] Mobile app
- [ ] Plugin ecosystem

---

**Author**: Dilip Reddy  
**Repository**: https://github.com/DilipReddy57/Annotix  
**Last Updated**: December 2024
