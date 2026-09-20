import numpy as np
from memory.buffer import EpisodicMemoryBuffer
from memory.utility_scorer import UtilityScorer
from agent.feature_extractor import FeatureExtractor

class RoboticAgent:
    """
    Simulates a long-horizon egocentric agent that uses task-conditioned memory.
    """
    def __init__(self, memory_capacity: int = 50, embedding_dim: int = 256):
        self.memory = EpisodicMemoryBuffer(capacity=memory_capacity)
        self.scorer = UtilityScorer(novelty_weight=1.0)
        self.feature_extractor = FeatureExtractor(embedding_dim=embedding_dim)
        self.time_step = 0

    def step(self, observation: np.ndarray) -> float:
        """
        Processes a single visual frame/observation.
        """
        # 1. Extract visual features
        emb = self.feature_extractor.extract(observation)
        
        # 2. Get recent temporal context to evaluate novelty/utility
        recent_context = self.memory.get_recent(n=5)
        
        # 3. Compute utility score
        score = self.scorer.compute_utility(emb, recent_context)
        
        # 4. Add to episodic memory
        memory_item = {
            'embedding': emb,
            'utility_score': score,
            'timestamp': self.time_step,
            'meta': {'observation_id': id(observation)}
        }
        self.memory.add(memory_item)
        
        self.time_step += 1
        return score
