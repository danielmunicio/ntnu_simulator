from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseVLAPlanner(ABC):
    """
    Base class for Vision-Language-Action (VLA) planners.
    Processes visual input and text prompts to output directional commands.
    Args:
        prompt: Text instruction or query
    """

    def __init__(self, prompt: str):
        """Initialize the VLA planner."""
        self.prompt = prompt

    @abstractmethod
    def get_direction(self, image: np.ndarray) -> Tuple[float, float, float]:
        """
        Process an image and text prompt to determine a direction to move.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Tuple of (x, y, z) representing the direction vector to move
        """
        raise NotImplementedError("Subclasses must implement get_direction method")
