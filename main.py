import numpy as np
from agent.robotic_agent import RoboticAgent

def main():
    print("Initializing Task-Conditioned Episodic Memory Agent...")
    # Small capacity to trigger evictions quickly in the simulation
    agent = RoboticAgent(memory_capacity=20, embedding_dim=128)
    
    num_steps = 100
    print(f"Simulating {num_steps} observation steps with a memory budget of {agent.memory.capacity}...")
    
    for t in range(num_steps):
        # Simulate a raw observation (e.g., an RGB image frame)
        dummy_observation = np.zeros((3, 224, 224)) 
        
        # The agent processes the observation and manages its memory budget internally
        utility_score = agent.step(dummy_observation)
        
        if (t + 1) % 25 == 0:
            print(f"\nStep {t+1}: Processed observation. Utility Score: {utility_score:.3f}")
            print(f"  Current Memory Size: {len(agent.memory.memory)} / {agent.memory.capacity}")
            print(f"  Total Evictions so far: {agent.memory.eviction_count}")

    print("\n--- Simulation Complete ---")
    print(f"Final Memory Size: {len(agent.memory.memory)}")
    print(f"Total items evicted to maintain budget: {agent.memory.eviction_count}")
    
    # Analyze what was kept
    if agent.memory.memory:
        retained_scores = [m['utility_score'] for m in agent.memory.memory]
        avg_retained_score = sum(retained_scores) / len(retained_scores)
        print(f"Average Utility Score of retained memories: {avg_retained_score:.3f}")
    
    print("\nNext Steps for the Team:")
    print("1. Replace FeatureExtractor with a real Vision model (e.g., CLIP, ViT).")
    print("2. Expand UtilityScorer to include LLM/VLA preference constraints.")
    print("3. Test with real Egocentric datasets like Ego4D.")

if __name__ == "__main__":
    main()
