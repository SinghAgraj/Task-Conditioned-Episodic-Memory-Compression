# Literature Review: Task-Conditioned Episodic Memory Compression for Egocentric Robotic Agents

## 1. Introduction
Long-horizon embodied agents operating in complex environments require the ability to remember past observations, interactions, and state changes to make informed decisions. However, egocentric visual streams (e.g., from head-mounted or robot-mounted cameras) contain significant redundancy. Storing raw visual streams indefinitely leads to "context bloat" and exceeds the bounded computation and persistent memory constraints of physical robots. Recent research has focused on **episodic memory compression** and **visual-history compression**, where information is selectively retained, summarized, or discarded based on its utility for current and future tasks. 

This review explores recent approaches to memory architecture, visual history compression, and task-aware retention strategies in robotic systems.

## 2. Hierarchical and Structured Memory Architectures
Traditional approaches often rely on flat, first-in-first-out (FIFO) sliding window buffers, which suffer from "catastrophic forgetting" of distant but critical events. Current state-of-the-art frameworks have moved toward hierarchical memory systems:
*   **Decoupled Frequencies:** Frameworks like **HiMe** propose separating agent cognitive processes into multiple frequencies. For example, a high-frequency "Executor" manages real-time control, a "Sentry" handles working memory, and a low-frequency "Planner" handles long-term strategic reasoning. This decoupling allows the agent to maintain an "infinite" structured memory without overwhelming the high-frequency control loop.
*   **Episodic vs. Semantic Memory:** Inspired by the human Entorhinal Cortex–Hippocampus–Prefrontal Cortex model, systems like **Chameleon** utilize differentiable memory stacks. They ground multimodal tokens in geometry and distinguish between short-term working memory (immediate context) and long-term episodic memory (past interactions that break perceptual aliasing).

## 3. Visual-History Compression Strategies
Processing long sequences of high-resolution images is computationally prohibitive for standard Vision-Language-Action (VLA) models. Researchers have developed strategies to compress visual history efficiently:
*   **Native Memory Compression:** Models such as **NativeMEM** encode historical visual frames directly into "native" memory tokens within the agent's own token space. By compressing frames into compact token representations, the system retains minute-level visual context without external vector databases or increasing the latency of the VLA model.
*   **Dual-Branch Processing:** Systems like **Remember Smarter (RS)** use a dual-branch approach. A fine-grained visual branch (often utilizing efficient architectures like spatial/temporal Mamba) compresses recent observations. Simultaneously, a coarse-grained branch stores successful past experiences in a specialized latent space (e.g., hyperbolic/Poincaré space) to facilitate the retrieval of distant, highly relevant memories.

## 4. Task-Utility-Aware and Preference-Conditioned Retention
Not all observations are equally important. Retaining information based on its task utility is a core challenge in memory allocation:
*   **Preference-Conditioned Compression:** Frameworks like **MeMento** selectively compress historical information based on user preferences and task relevance. By focusing solely on "decision-relevant" data, these systems can reduce memory usage by over 80% while simultaneously improving task accuracy, as the model is not distracted by irrelevant visual noise.
*   **Sparse Thinking and Context Folding:** Some approaches utilize reinforcement fine-tuning to refine the agent's thought-action chains. "Sparse Thinking" involves the agent summarizing or "folding" past sub-trajectories into abstract representations once specific sub-task milestones are reached. The memory buffer is only updated with these condensed milestones, rather than continuous visual frames.

## 5. Conclusion and Implications for the Project
The fundamental challenge in episodic memory for robotics is the **"frequency-competence paradox"**: high-capacity models are too slow for real-time control, while fast models lack the context for long-horizon planning. 
For this project, investigating a **task-utility-aware allocation scheme** is highly relevant. By selectively retaining observations based on novelty, interaction changes, and future task utility, the proposed framework can bridge the gap between memory consumption and long-horizon decision-making performance. Developing a metric for "task utility" (e.g., surprise, semantic change, milestone completion) will be a critical step in building the compression framework.

## References
1. Research on "NativeMEM" and VLA token-space compression.
2. The "HiMe" hierarchical memory framework for infinite context handling.
3. Dual-branch memory modules (e.g., "Remember Smarter").
4. "MeMento" and preference-conditioned compression frameworks.
5. "Sparse Thinking" and context folding in reinforcement fine-tuned agents.
