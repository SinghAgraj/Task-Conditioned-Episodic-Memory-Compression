import numpy as np

class UtilityScorer:
    """
    Scores incoming observations to determine their retention utility based on novelty, 
    interaction changes, and task relevance.
    """
    def __init__(self, novelty_weight: float = 1.0):
        self.novelty_weight = novelty_weight

    def compute_utility(self, current_embedding: np.ndarray, recent_memories: list) -> float:
        """
        Computes the utility of the current observation.
        High utility = novel (low similarity to recent memories) or significant state change.
        """
        if not recent_memories:
            return 1.0  # High utility for the very first observation
            
        recent_embs = np.array([m['embedding'] for m in recent_memories])
        
        # Compute cosine similarity
        dot_products = np.dot(recent_embs, current_embedding)
        norms = np.linalg.norm(recent_embs, axis=1) * np.linalg.norm(current_embedding)
        similarities = dot_products / (norms + 1e-8)
        
        # Maximum similarity to any recent memory
        max_sim = np.max(similarities)
        
        # Novelty is inversely proportional to similarity
        novelty = 1.0 - max_sim
        
        # Final utility score (bounded 0 to 1)
        utility_score = novelty * self.novelty_weight
        return float(np.clip(utility_score, 0.0, 1.0))
