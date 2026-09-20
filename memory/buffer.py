import numpy as np
from typing import Dict, Any, List

class EpisodicMemoryBuffer:
    """
    A fixed-size episodic memory buffer that retains observations based on task utility.
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        # Stores dictionaries with keys: 'embedding', 'utility_score', 'timestamp', 'meta'
        self.memory: List[Dict[str, Any]] = []
        self.eviction_count = 0

    def add(self, memory_item: Dict[str, Any]):
        """
        Adds a new memory item. If capacity is reached, evicts the lowest utility item.
        """
        if len(self.memory) >= self.capacity:
            self._evict_lowest_utility()
        self.memory.append(memory_item)

    def _evict_lowest_utility(self):
        """
        Finds and removes the memory item with the lowest utility score.
        """
        if not self.memory:
            return
        
        # Find index of item with lowest utility
        lowest_idx = min(range(len(self.memory)), key=lambda i: self.memory[i].get('utility_score', 0))
        evicted = self.memory.pop(lowest_idx)
        self.eviction_count += 1
        # print(f"Evicted memory t={evicted['timestamp']} (score: {evicted['utility_score']:.3f})")

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Returns the n most recent memories to provide temporal context."""
        return self.memory[-n:]
    
    def get_all_embeddings(self) -> np.ndarray:
        if not self.memory:
            return np.array([])
        return np.array([m['embedding'] for m in self.memory])
