from typing import Any
from backend.core.logger import get_logger

logger = get_logger("synthetic")

class SyntheticDataAgent:
    def generate_variations(self, image: Any, mask: Any) -> Any:
        """
        Generates synthetic training data by augmenting existing images and masks.
        """
        logger.info("Generating synthetic variations...")
        # Mock augmentation
        return [
            {"type": "flip_horizontal", "image": "mock_img", "mask": "mock_mask"},
            {"type": "rotate_90", "image": "mock_img", "mask": "mock_mask"}
        ]
