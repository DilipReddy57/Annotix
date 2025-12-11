"""
Feedback API Routes for User Corrections and Learning.

Provides endpoints for:
- Recording label corrections
- Approving/rejecting labels
- Getting suggestions based on learned patterns
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlmodel import Session

from backend.core.database import get_session
from backend.core.logger import get_logger
from backend.agents.instance_learner import InstanceLearner
from backend.agents.context_learner import ContextLearner

router = APIRouter()
logger = get_logger("feedback")

# Initialize learning agents
instance_learner = InstanceLearner()
context_learner = ContextLearner()


# Request/Response Models
class CorrectionRequest(BaseModel):
    original_label: str
    corrected_label: str
    image_id: str
    project_id: str
    bbox: Optional[List[float]] = None
    embedding: Optional[List[float]] = None


class ApprovalRequest(BaseModel):
    label: str
    project_id: str


class DomainDetectionRequest(BaseModel):
    labels: List[str]
    project_id: Optional[str] = None


class DomainSetRequest(BaseModel):
    project_id: str
    domain: str


# Endpoints
@router.post("/correction")
async def record_correction(request: CorrectionRequest):
    """
    Record a user label correction for learning.
    
    The system will:
    1. Learn the mapping (original -> corrected)
    2. Find similar annotations that might need same correction
    3. Boost confidence for future detections
    """
    import numpy as np
    
    embedding = None
    if request.embedding:
        embedding = np.array(request.embedding)
    
    result = instance_learner.record_correction(
        original_label=request.original_label,
        corrected_label=request.corrected_label,
        image_id=request.image_id,
        project_id=request.project_id,
        embedding=embedding,
        bbox=request.bbox
    )
    
    logger.info(f"Correction recorded: {request.original_label} -> {request.corrected_label}")
    
    return result


@router.post("/approve")
async def approve_label(request: ApprovalRequest):
    """Record that a label was approved by user."""
    instance_learner.record_approval(request.label, request.project_id)
    
    return {
        "status": "approved",
        "label": request.label,
        "confidence_adjustment": instance_learner.get_confidence_adjustment(request.label)
    }


@router.post("/reject")
async def reject_label(request: ApprovalRequest):
    """Record that a label was rejected by user."""
    instance_learner.record_rejection(request.label, request.project_id)
    
    return {
        "status": "rejected",
        "label": request.label,
        "confidence_adjustment": instance_learner.get_confidence_adjustment(request.label)
    }


@router.get("/mappings")
async def get_learned_mappings(project_id: Optional[str] = None):
    """Get all learned label mappings."""
    mappings = instance_learner.get_learned_mappings(project_id)
    
    return {
        "mappings": mappings,
        "count": len(mappings)
    }


@router.get("/statistics")
async def get_learning_statistics():
    """Get learning statistics."""
    return instance_learner.get_statistics()


# Context/Domain endpoints
@router.post("/domain/detect")
async def detect_domain(request: DomainDetectionRequest):
    """
    Detect the most likely domain based on labels.
    
    Useful for auto-configuring annotation settings.
    """
    result = context_learner.detect_domain(
        labels=request.labels,
        project_id=request.project_id
    )
    
    return result


@router.post("/domain/set")
async def set_project_domain(request: DomainSetRequest):
    """Explicitly set a project's domain."""
    context_learner.set_project_domain(request.project_id, request.domain)
    
    return {
        "status": "set",
        "project_id": request.project_id,
        "domain": request.domain
    }


@router.get("/domain/profiles")
async def list_domain_profiles():
    """List all available domain profiles."""
    profiles = {}
    for name, profile in context_learner.profiles.items():
        profiles[name] = {
            "name": profile.name,
            "description": profile.description,
            "expected_labels_count": len(profile.expected_labels),
            "sample_prompts": profile.suggested_prompts[:5]
        }
    
    return {
        "profiles": profiles,
        "count": len(profiles)
    }


@router.get("/suggestions/prompts")
async def get_prompt_suggestions(
    project_id: Optional[str] = None,
    domain: Optional[str] = None,
    current_labels: Optional[str] = None
):
    """
    Get suggested prompts based on context.
    
    Args:
        project_id: Project to get suggestions for
        domain: Explicit domain to use
        current_labels: Comma-separated list of already detected labels
    """
    labels = current_labels.split(",") if current_labels else None
    
    prompts = context_learner.get_suggested_prompts(
        project_id=project_id,
        domain=domain,
        current_labels=labels
    )
    
    return {
        "suggestions": prompts,
        "domain": domain or context_learner.project_domains.get(project_id, "general")
    }


@router.get("/suggestions/relabel/{label}")
async def get_relabel_suggestion(label: str):
    """
    Get suggested relabeling based on learned corrections.
    
    Args:
        label: Label to check for corrections
    """
    corrected, boost = instance_learner.apply_learned_correction(label)
    
    return {
        "original": label,
        "suggested": corrected,
        "has_correction": corrected != label,
        "confidence_boost": boost
    }
