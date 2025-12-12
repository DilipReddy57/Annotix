"""
Context-Aware Learning Agent for Domain-Specific Annotation Intelligence.

This agent provides:
- Domain profiles (autonomous driving, medical, retail, etc.)
- Context-based prompt suggestions
- Domain auto-detection from annotations
- Project-specific learning

Features:
- Pre-built domain ontologies
- Auto-detect domain from first annotations
- Suggest prompts based on context
- Learn new domains from user input
"""

import os
import json
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

from backend.core.logger import get_logger

logger = get_logger("context_learner")


@dataclass
class DomainProfile:
    """A domain profile with expected labels and prompts."""
    name: str
    description: str
    expected_labels: List[str]
    suggested_prompts: List[str]
    parent_categories: Dict[str, List[str]] = field(default_factory=dict)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pre-built domain profiles
DOMAIN_PROFILES = {
    "autonomous_driving": DomainProfile(
        name="Autonomous Driving",
        description="Self-driving car perception - vehicles, pedestrians, traffic",
        expected_labels=[
            "car", "truck", "bus", "motorcycle", "bicycle",
            "pedestrian", "person", "cyclist",
            "traffic_light", "stop_sign", "speed_limit", "road_sign",
            "lane_marking", "crosswalk", "road", "sidewalk",
            "traffic_cone", "barrier"
        ],
        suggested_prompts=[
            "car", "truck", "bus", "motorcycle", "bicycle",
            "pedestrian", "person crossing street",
            "traffic light", "stop sign", "road sign",
            "lane markings", "crosswalk"
        ],
        parent_categories={
            "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "person": ["pedestrian", "cyclist", "person"],
            "traffic_control": ["traffic_light", "stop_sign", "speed_limit", "road_sign"],
            "road_infrastructure": ["lane_marking", "crosswalk", "road", "sidewalk"]
        }
    ),
    
    "medical_imaging": DomainProfile(
        name="Medical Imaging",
        description="Medical scans, X-rays, pathology - organs, lesions, anatomy",
        expected_labels=[
            "tumor", "lesion", "nodule", "cyst",
            "organ", "heart", "lung", "liver", "kidney", "brain",
            "bone", "fracture", "tissue", "blood_vessel",
            "abnormality", "inflammation"
        ],
        suggested_prompts=[
            "tumor", "lesion", "nodule",
            "organ", "heart", "lung", "liver",
            "bone", "fracture",
            "abnormality", "mass"
        ],
        parent_categories={
            "pathology": ["tumor", "lesion", "nodule", "cyst", "abnormality"],
            "organ": ["heart", "lung", "liver", "kidney", "brain"],
            "anatomy": ["bone", "tissue", "blood_vessel"]
        }
    ),
    
    "retail": DomainProfile(
        name="Retail & E-commerce",
        description="Product detection, shelf analysis, inventory",
        expected_labels=[
            "product", "shelf", "price_tag", "barcode",
            "bottle", "can", "box", "package",
            "customer", "shopping_cart", "basket",
            "checkout", "display", "promotion"
        ],
        suggested_prompts=[
            "product on shelf", "price tag", "barcode",
            "bottle", "can", "box",
            "customer", "shopping cart",
            "promotional display"
        ],
        parent_categories={
            "product": ["bottle", "can", "box", "package", "product"],
            "infrastructure": ["shelf", "checkout", "display"],
            "customer": ["customer", "shopping_cart", "basket"]
        }
    ),
    
    "agriculture": DomainProfile(
        name="Agriculture & Farming",
        description="Crop monitoring, livestock, farm equipment",
        expected_labels=[
            "crop", "plant", "weed", "pest", "disease",
            "fruit", "vegetable", "leaf", "stem", "flower",
            "livestock", "cow", "sheep", "pig",
            "tractor", "irrigation", "soil"
        ],
        suggested_prompts=[
            "crop", "plant", "weed", "pest",
            "fruit", "vegetable",
            "livestock animal",
            "farming equipment", "tractor"
        ],
        parent_categories={
            "crop": ["plant", "fruit", "vegetable", "leaf", "stem", "flower"],
            "problem": ["weed", "pest", "disease"],
            "livestock": ["cow", "sheep", "pig"],
            "equipment": ["tractor", "irrigation"]
        }
    ),
    
    "construction": DomainProfile(
        name="Construction & Safety",
        description="Construction sites, safety equipment, workers",
        expected_labels=[
            "worker", "hard_hat", "safety_vest", "scaffold",
            "crane", "excavator", "bulldozer",
            "concrete", "steel", "beam", "pipe",
            "hazard", "safety_zone", "warning_sign"
        ],
        suggested_prompts=[
            "construction worker", "hard hat", "safety vest",
            "crane", "excavator", "heavy equipment",
            "scaffolding", "hazard",
            "warning sign"
        ],
        parent_categories={
            "person": ["worker"],
            "safety": ["hard_hat", "safety_vest", "safety_zone", "warning_sign"],
            "equipment": ["crane", "excavator", "bulldozer", "scaffold"],
            "material": ["concrete", "steel", "beam", "pipe"]
        }
    ),
    
    "wildlife": DomainProfile(
        name="Wildlife & Nature",
        description="Animal detection, species identification, conservation",
        expected_labels=[
            "animal", "bird", "mammal", "reptile", "fish",
            "deer", "bear", "wolf", "fox", "rabbit",
            "eagle", "owl", "duck", "heron",
            "tree", "vegetation", "water", "nest"
        ],
        suggested_prompts=[
            "animal", "bird", "mammal",
            "deer", "bear", "wolf",
            "eagle", "owl",
            "wildlife"
        ],
        parent_categories={
            "mammal": ["deer", "bear", "wolf", "fox", "rabbit"],
            "bird": ["eagle", "owl", "duck", "heron"],
            "animal": ["mammal", "bird", "reptile", "fish"],
            "environment": ["tree", "vegetation", "water", "nest"]
        }
    ),
    
    "general": DomainProfile(
        name="General Purpose",
        description="Common objects for general annotation tasks",
        expected_labels=[
            "person", "car", "dog", "cat", "chair", "table",
            "bottle", "cup", "phone", "laptop", "book",
            "tree", "building", "sky", "ground"
        ],
        suggested_prompts=[
            "person", "people", "car", "vehicle",
            "animal", "dog", "cat",
            "furniture", "object"
        ],
        parent_categories={
            "person": ["person", "people"],
            "vehicle": ["car", "truck", "bus"],
            "animal": ["dog", "cat"],
            "object": ["chair", "table", "bottle", "cup", "phone"]
        }
    )
}


class ContextLearner:
    """
    Context-Aware Learning Agent that adapts to project domains.
    
    Features:
    - Auto-detect domain from annotations
    - Suggest context-appropriate prompts
    - Learn custom domains from user feedback
    """
    
    def __init__(self, custom_domains_path: Optional[str] = None):
        """
        Initialize Context Learner.
        
        Args:
            custom_domains_path: Path to custom domain definitions JSON
        """
        self.profiles: Dict[str, DomainProfile] = dict(DOMAIN_PROFILES)
        self.project_domains: Dict[str, str] = {}  # project_id -> domain_name
        self.project_labels: Dict[str, Counter] = defaultdict(Counter)
        
        # Load custom domains if provided
        if custom_domains_path and os.path.exists(custom_domains_path):
            self._load_custom_domains(custom_domains_path)
        
        logger.info(f"ContextLearner initialized with {len(self.profiles)} domain profiles")
    
    def _load_custom_domains(self, path: str):
        """Load custom domain definitions from JSON file."""
        try:
            with open(path, 'r') as f:
                custom = json.load(f)
            
            for name, data in custom.items():
                self.profiles[name] = DomainProfile(
                    name=data.get("name", name),
                    description=data.get("description", ""),
                    expected_labels=data.get("expected_labels", []),
                    suggested_prompts=data.get("suggested_prompts", []),
                    parent_categories=data.get("parent_categories", {}),
                    confidence=data.get("confidence", 0.8)
                )
            logger.info(f"Loaded {len(custom)} custom domain profiles")
        except Exception as e:
            logger.error(f"Failed to load custom domains: {e}")
    
    def detect_domain(
        self, 
        labels: List[str], 
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect the most likely domain based on observed labels.
        
        Args:
            labels: List of labels detected so far
            project_id: Optional project to remember domain for
            
        Returns:
            Dict with detected domain and confidence
        """
        if not labels:
            return {"domain": "general", "confidence": 0.0, "profile": self.profiles["general"].to_dict()}
        
        label_set = set(l.lower() for l in labels)
        scores = {}
        
        for domain_name, profile in self.profiles.items():
            expected_set = set(l.lower() for l in profile.expected_labels)
            overlap = label_set & expected_set
            
            if expected_set:
                # Score based on overlap ratio
                score = len(overlap) / len(expected_set)
                # Boost score if most detected labels match
                coverage = len(overlap) / max(len(label_set), 1)
                scores[domain_name] = (score + coverage) / 2
            else:
                scores[domain_name] = 0.0
        
        # Find best match
        best_domain = max(scores, key=scores.get)
        confidence = scores[best_domain]
        
        # Remember for project
        if project_id and confidence > 0.3:
            self.project_domains[project_id] = best_domain
            self.project_labels[project_id].update(labels)
        
        return {
            "domain": best_domain,
            "confidence": confidence,
            "profile": self.profiles[best_domain].to_dict(),
            "all_scores": scores
        }
    
    def get_suggested_prompts(
        self, 
        project_id: Optional[str] = None,
        domain: Optional[str] = None,
        current_labels: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get suggested prompts based on context.
        
        Args:
            project_id: Project to get suggestions for
            domain: Explicit domain to use
            current_labels: Labels already detected
            
        Returns:
            List of suggested prompts
        """
        # Use project's remembered domain
        if project_id and project_id in self.project_domains:
            domain = self.project_domains[project_id]
        
        # Default to general
        if not domain or domain not in self.profiles:
            domain = "general"
        
        profile = self.profiles[domain]
        prompts = list(profile.suggested_prompts)
        
        # Filter out already detected labels
        if current_labels:
            detected_lower = set(l.lower() for l in current_labels)
            prompts = [p for p in prompts if p.lower() not in detected_lower]
        
        return prompts[:10]  # Return top 10 suggestions
    
    def get_parent_category(self, label: str, domain: Optional[str] = None) -> Optional[str]:
        """
        Get the parent category for a label.
        
        Args:
            label: Label to categorize
            domain: Domain to search in
            
        Returns:
            Parent category or None
        """
        domains_to_check = [domain] if domain else list(self.profiles.keys())
        label_lower = label.lower()
        
        for d in domains_to_check:
            if d not in self.profiles:
                continue
            for parent, children in self.profiles[d].parent_categories.items():
                if label_lower in [c.lower() for c in children]:
                    return parent
        
        return None
    
    def set_project_domain(self, project_id: str, domain: str):
        """Explicitly set a project's domain."""
        if domain in self.profiles:
            self.project_domains[project_id] = domain
            logger.info(f"Set project {project_id} domain to {domain}")
        else:
            logger.warning(f"Unknown domain: {domain}")
    
    def add_custom_domain(
        self,
        name: str,
        description: str,
        expected_labels: List[str],
        suggested_prompts: Optional[List[str]] = None,
        parent_categories: Optional[Dict[str, List[str]]] = None
    ):
        """
        Add a new custom domain profile.
        
        Args:
            name: Domain identifier
            description: Human-readable description
            expected_labels: List of expected labels
            suggested_prompts: Optional prompt suggestions
            parent_categories: Optional category hierarchy
        """
        self.profiles[name] = DomainProfile(
            name=name.replace("_", " ").title(),
            description=description,
            expected_labels=expected_labels,
            suggested_prompts=suggested_prompts or expected_labels[:10],
            parent_categories=parent_categories or {},
            confidence=0.9
        )
        logger.info(f"Added custom domain: {name}")
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about domains and projects."""
        return {
            "total_profiles": len(self.profiles),
            "profiles": list(self.profiles.keys()),
            "project_domains": dict(self.project_domains),
            "project_label_counts": {
                pid: dict(counter) for pid, counter in self.project_labels.items()
            }
        }
    
    def export_profiles(self, output_path: str):
        """Export all domain profiles to JSON."""
        data = {
            name: profile.to_dict() 
            for name, profile in self.profiles.items()
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} profiles to {output_path}")
