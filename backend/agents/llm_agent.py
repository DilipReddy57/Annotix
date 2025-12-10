"""
LLM Agent for intelligent prompt generation and refinement.

Features:
- Auto-generate optimal prompts for SAM3 from image content
- Refine and enhance user prompts for better detection
- Multi-modal understanding using vision-language models
- Context-aware annotation suggestions
"""

import os
import base64
import json
from typing import Dict, Any, List, Optional
from PIL import Image
import io

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("llm_agent")


class LLMAgent:
    """
    LLM Agent for intelligent annotation assistance.
    
    Uses Google Gemini or OpenAI GPT for:
    - Analyzing images to suggest prompts
    - Refining user prompts for better SAM3 detection
    - Generating descriptions for annotations
    - Quality assessment of annotations
    """
    
    def __init__(self, provider: str = "gemini"):
        """
        Initialize LLM Agent.
        
        Args:
            provider: "gemini" or "openai"
        """
        self.provider = provider
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the LLM client."""
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        else:
            logger.warning(f"Unknown provider: {self.provider}, using mock mode")
    
    def _init_gemini(self):
        """Initialize Google Gemini."""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.0-flash")
                logger.info("✅ Gemini initialized successfully")
            else:
                logger.warning("No GEMINI_API_KEY found, LLM features disabled")
                
        except ImportError:
            logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    def _init_openai(self):
        """Initialize OpenAI."""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model = "gpt-4o"
                logger.info("✅ OpenAI initialized successfully")
            else:
                logger.warning("No OPENAI_API_KEY found")
                
        except ImportError:
            logger.warning("openai not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for API calls."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def analyze_image_for_prompts(
        self, 
        image_path: str,
        context: Optional[str] = None,
        max_prompts: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Analyze an image and generate optimal SAM3 prompts.
        
        Args:
            image_path: Path to the image
            context: Optional context (e.g., "autonomous driving", "medical imaging")
            max_prompts: Maximum number of prompts to generate
            
        Returns:
            List of suggested prompts with confidence and reasoning
        """
        if self.model is None:
            return self._mock_prompts(image_path)
        
        prompt = f"""Analyze this image and generate {max_prompts} optimal text prompts for SAM 3 
(Segment Anything Model 3) to detect and segment objects.

{f'Context: {context}' if context else ''}

For each detected object or concept, provide:
1. The exact prompt text (be specific, e.g., "red car on the left" not just "car")
2. Confidence score (0-1) that this object exists in the image
3. Brief reasoning

Return as JSON array:
[
  {{"prompt": "specific object description", "confidence": 0.95, "reason": "clearly visible in center"}},
  ...
]

Focus on:
- Specific attributes (colors, positions, sizes)
- Distinguishing features ("person wearing blue jacket")
- Spatial relationships ("laptop on wooden desk")
- All visible objects, foreground and background
"""
        
        try:
            if self.provider == "gemini":
                return self._analyze_with_gemini(image_path, prompt)
            elif self.provider == "openai":
                return self._analyze_with_openai(image_path, prompt)
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._mock_prompts(image_path)
    
    def _analyze_with_gemini(self, image_path: str, prompt: str) -> List[Dict[str, Any]]:
        """Analyze image using Gemini."""
        import google.generativeai as genai
        
        img = Image.open(image_path)
        response = self.model.generate_content([prompt, img])
        
        # Parse JSON from response
        text = response.text
        # Extract JSON from markdown code block if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
    
    def _analyze_with_openai(self, image_path: str, prompt: str) -> List[Dict[str, Any]]:
        """Analyze image using OpenAI."""
        base64_image = self._image_to_base64(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        text = response.choices[0].message.content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
            
        return json.loads(text)
    
    def refine_prompt(
        self, 
        user_prompt: str, 
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine a user prompt for better SAM3 detection.
        
        Args:
            user_prompt: Original user prompt
            image_path: Optional image for context
            
        Returns:
            Dict with refined_prompt, improvements, and reasoning
        """
        if self.model is None:
            return {
                "original": user_prompt,
                "refined": user_prompt,
                "improvements": [],
                "reasoning": "LLM not available"
            }
        
        system_prompt = """You are an expert at crafting prompts for SAM 3 (Segment Anything Model 3).
        
SAM 3 works best with:
- Specific object descriptions (not generic)
- Color and attribute mentions
- Spatial context when multiple similar objects exist
- Singular nouns for individual instances

Given a user prompt, refine it for optimal SAM3 detection.
Return JSON:
{
    "original": "user prompt",
    "refined": "improved prompt",
    "improvements": ["added color", "added position"],
    "reasoning": "why these changes help"
}
"""
        
        try:
            if self.provider == "gemini":
                if image_path:
                    img = Image.open(image_path)
                    response = self.model.generate_content([
                        system_prompt,
                        f"User prompt: {user_prompt}",
                        img
                    ])
                else:
                    response = self.model.generate_content([
                        system_prompt,
                        f"User prompt: {user_prompt}"
                    ])
                
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                return json.loads(text)
                
        except Exception as e:
            logger.error(f"Prompt refinement failed: {e}")
            return {
                "original": user_prompt,
                "refined": user_prompt,
                "improvements": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    def generate_annotation_description(
        self,
        image_path: str,
        bbox: List[int],
        label: str
    ) -> str:
        """
        Generate a detailed description for an annotation.
        
        Args:
            image_path: Path to the image
            bbox: Bounding box [x, y, w, h]
            label: Detection label
            
        Returns:
            Detailed description of the annotated object
        """
        if self.model is None:
            return f"A {label} detected in the image."
        
        try:
            img = Image.open(image_path)
            
            # Crop to bbox
            x, y, w, h = bbox
            cropped = img.crop((x, y, x + w, y + h))
            
            prompt = f"""Describe this object (labeled as "{label}") in detail.
Include: appearance, color, condition, any notable features.
Keep it to 1-2 sentences."""
            
            if self.provider == "gemini":
                response = self.model.generate_content([prompt, cropped])
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            
        return f"A {label} detected in the image."
    
    def assess_annotation_quality(
        self,
        image_path: str,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to assess the quality of annotations.
        
        Args:
            image_path: Path to the annotated image
            annotations: List of annotations with labels and bboxes
            
        Returns:
            Quality assessment with suggestions
        """
        if self.model is None:
            return {"quality": "unknown", "suggestions": [], "missing": []}
        
        ann_summary = json.dumps([
            {"label": a.get("label"), "bbox": a.get("bbox")}
            for a in annotations
        ], indent=2)
        
        prompt = f"""Review these annotations for an image:

{ann_summary}

Assess:
1. Are there likely missing objects?
2. Are labels appropriate and specific enough?
3. Any quality concerns?

Return JSON:
{{
    "quality": "good/fair/poor",
    "score": 0.0-1.0,
    "suggestions": ["suggestion 1", ...],
    "missing": ["possibly missing object 1", ...]
}}"""
        
        try:
            img = Image.open(image_path)
            
            if self.provider == "gemini":
                response = self.model.generate_content([prompt, img])
                text = response.text
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                return json.loads(text)
                
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            
        return {"quality": "unknown", "suggestions": [], "missing": []}
    
    def _mock_prompts(self, image_path: str) -> List[Dict[str, Any]]:
        """Return mock prompts when LLM is not available."""
        return [
            {"prompt": "main subject in foreground", "confidence": 0.9, "reason": "Primary focus of image"},
            {"prompt": "background elements", "confidence": 0.7, "reason": "Scene context"},
            {"prompt": "all visible objects", "confidence": 0.95, "reason": "Exhaustive detection"}
        ]
