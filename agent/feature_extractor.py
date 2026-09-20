import numpy as np
from typing import Any

class FeatureExtractor:
    """
    Mock feature extractor representing a CNN or Vision Transformer (e.g. CLIP/ViT).
    In a real project, this would take an image and return a dense embedding vector.
    """
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim

    def extract(self, observation: Any) -> np.ndarray:
        """
        Takes a raw observation (e.g. image array) and returns a normalized embedding.
        For simulation, we generate a random vector.
        """
        # In reality: return self.model(observation)
        # We simulate this by returning a random normalized vector
        vec = np.random.randn(self.embedding_dim)
        return vec / np.linalg.norm(vec)
