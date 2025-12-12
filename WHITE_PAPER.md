# ANNOTIX: Autonomous AI Annotation Platform

## Technical White Paper v2.0

---

## Executive Summary

ANNOTIX is an autonomous image/video annotation platform that replaces manual pixel-by-pixel labeling with intelligent reasoning. By combining **SAM3** (Segment Anything Model 3), **CLIP** (real visual embeddings), **RAG** (knowledge persistence), and **LLM** (scene analysis), we achieve:

- **Fast by default**: ~3-5 seconds per image on RTX 3060
- **Turbo mode**: <1 second per image (reduced features)
- **Honest AI**: Real embeddings, real consistency, no fake features

---

## 1. The Problem We Solve

**Traditional Annotation Tools** (CVAT, LabelImg, Roboflow):

- Manual: Draw every bounding box by hand
- Slow: 2-5+ minutes per complex image
- Inconsistent: Human fatigue causes labeling errors
- Stateless: Each image starts from scratch

**ANNOTIX Difference**:

- Autonomous: AI identifies and segments objects
- Fast: Single SAM3 inference + batch CLIP embeddings
- Consistent: Real vector similarity ensures label uniformity
- Learning: RAG database remembers past annotations

---

## 2. Architecture: What's Actually Running

### Core Pipeline (`smart_process_image`)

```
┌─────────────────────────────────────────────────────────────────┐
│                      ANNOTIX PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Image file                                              │
│    ↓                                                            │
│  [STEP 1] Prompt Generation (LLM)          ~500-1000ms          │
│    • Gemini analyzes image                                      │
│    • Returns: ["person", "car", "traffic light"]                │
│    • TURBO MODE: Skip this, use "all objects"                   │
│    ↓                                                            │
│  [STEP 2] SAM3 Segmentation                ~2000-4000ms         │
│    • Single inference with best prompt                          │
│    • Returns: masks, bboxes, confidence scores                  │
│    ↓                                                            │
│  [STEP 3] Mask Refinement                  ~50ms                │
│    • Clean edges with morphological ops                         │
│    ↓                                                            │
│  [STEP 4] CLIP Embeddings (REAL)           ~100-200ms           │
│    • Batch extract from all detected regions                    │
│    • 512-dim semantic vectors                                   │
│    • TURBO MODE: Skip this                                      │
│    ↓                                                            │
│  [STEP 5] RAG Consistency                  ~50ms                │
│    • Query ChromaDB for similar past annotations                │
│    • Weighted voting for label consistency                      │
│    • TURBO MODE: Just normalize label                           │
│    ↓                                                            │
│  [STEP 6] QA Validation                    ~10ms                │
│    • Size sanity checks                                         │
│    • Overlap detection                                          │
│    ↓                                                            │
│  [STEP 7] Deduplication                    ~10ms                │
│    • Remove overlapping annotations (IoU threshold)             │
│    ↓                                                            │
│  [STEP 8] Background Analytics             (non-blocking)       │
│    • Store embeddings in RAG                                    │
│    • Update UMAP visualization                                  │
│    • Track for active learning                                  │
│    ↓                                                            │
│  OUTPUT: Annotations + scene_graph + timing                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Processing Modes

| Mode               | LLM Prompts | CLIP Embeddings | RAG Lookup     | Estimated Time |
| ------------------ | ----------- | --------------- | -------------- | -------------- |
| **Fast** (default) | 1 prompt    | Real            | Yes            | 3-5 seconds    |
| **Turbo**          | Skip        | Skip            | Normalize only | <1 second      |

---

## 3. What Makes This Honest

### Real Embeddings (Not Random Noise)

```python
# OLD (fake):
embedding = np.random.rand(256)  # Useless for similarity

# NEW (real):
from backend.agents.embedding_extractor import get_embedding_extractor
extractor = get_embedding_extractor()
embedding = extractor.extract_embedding(image, bbox)  # CLIP 512-dim
```

### Real RAG Consistency

When you label "excavator" in image 1, and image 100 has a similar yellow machine:

1. CLIP extracts embedding of the new object
2. ChromaDB finds the 5 most similar past objects
3. Weighted voting determines: "Previous similar objects were labeled 'excavator'"
4. System applies consistent label

### Real Timing

Every response includes `processing_time_ms`:

```json
{
  "status": "success",
  "processing_time_ms": 3847,
  "processing_mode": "fast",
  "annotations": [...]
}
```

---

## 4. API Endpoints

### Annotate Image

```
POST /api/projects/{project_id}/images/{image_id}/annotate?turbo_mode=false
```

### Test Pipeline (for verification)

```
POST /api/system/test-pipeline?image_url=https://example.com/cat.jpg&turbo_mode=false
```

Returns:

```json
{
  "status": "success",
  "mode": "fast",
  "processing_time_ms": 4231,
  "annotations_count": 3,
  "annotations": [
    { "label": "cat", "confidence": 0.94, "bbox": [100, 50, 200, 180] }
  ]
}
```

---

## 5. Requirements & Performance

### Hardware

- **Minimum**: 8GB RAM, GTX 1060 (6GB VRAM)
- **Recommended**: 16GB RAM, RTX 3060 (12GB VRAM)
- **Optimal**: 32GB RAM, RTX 4080+ (16GB+ VRAM)

### Expected Performance

| GPU      | Fast Mode | Turbo Mode |
| -------- | --------- | ---------- |
| GTX 1060 | 8-12s     | 2-4s       |
| RTX 3060 | 3-5s      | <1s        |
| RTX 4080 | 1-2s      | <0.5s      |

---

## 6. Deployed Features

| Feature            | Status    | Notes                       |
| ------------------ | --------- | --------------------------- |
| SAM3 Segmentation  | ✅ Active | 848M parameter model        |
| CLIP Embeddings    | ✅ Active | Real visual similarity      |
| RAG Consistency    | ✅ Active | ChromaDB persistent storage |
| LLM Auto-Prompts   | ✅ Active | Gemini integration          |
| QA Validation      | ✅ Active | Size/overlap checks         |
| Scene Graphs       | ✅ Active | Spatial relationships       |
| UMAP Visualization | ✅ Active | Embedding clusters          |
| Active Learning    | ✅ Active | Sample prioritization       |
| Turbo Mode         | ✅ Active | Maximum speed option        |

---

## 7. Conclusion

ANNOTIX delivers on its promises:

- **Speed**: Verified with `processing_time_ms` in every response
- **Accuracy**: Real CLIP embeddings + RAG voting
- **Consistency**: Persistent knowledge across annotations
- **Transparency**: No fake features, honest metrics

The system is production-ready with clear performance expectations based on your hardware.

---

_Last Updated: December 2024_
_Repository: https://github.com/DilipReddy57/Annotix_
