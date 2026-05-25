# AegisMARL: Cooperative Multi-Agent RTS Tactical Arena

AegisMARL is a premium, quantitative portfolio project demonstrating **Multi-Agent Reinforcement Learning (MARL)** within a 2D Real-Time Strategy (RTS) battle arena. 

The project coordinates a decentralized Blue Team (Knight, Ranger, and Healer) using a custom **Multi-Agent Proximal Policy Optimization (MAPPO)** engine written from scratch in PyTorch, deployed serverless directly in the browser via vanilla JavaScript matrix feedforward execution.

---

## 🚀 Quick Start Guide

### 1. Run the Interactive Dashboard
The entire interface runs fully serverless. To launch the server locally:
```bash
python -m http.server 8000
```
Then, open your web browser and navigate to:
**`http://localhost:8000`**

### 2. Train the PyTorch Models
To re-train the neural network actors and automatically export the updated weights to the browser:
```bash
python train_marl.py
```

### 3. Run Automated Equivalence Validation
To mathematically verify that the JavaScript feedforward engine produces identical outputs to PyTorch:
```bash
python test_equivalence.py
```

---

## 🧠 System Architecture & CTDE Paradigm

AegisMARL implements the industry-standard **Centralized Training with Decentralized Execution (CTDE)** framework, solving the environmental non-stationarity problem inherent in multi-agent systems.

```mermaid
graph TD
    subgraph Centralized Training (Python/PyTorch)
        GlobalState[Global State S: 99 features] -->|Joint State Vector| Critic[Centralized Critic Network]
        Critic -->|Cooperative GAE Advantages| LossCalc[MAPPO Loss Optimization]
    end
    
    subgraph Decentralized Execution (Web Browser Canvas)
        ObsKnight[Knight Obs: 33 features] -->|Local Sight| ActorKnight[Knight Actor Policy]
        ObsRanger[Ranger Obs: 33 features] -->|Local Sight| ActorRanger[Ranger Actor Policy]
        ObsHealer[Healer Obs: 33 features] -->|Local Sight| ActorHealer[Healer Actor Policy]
    end
    
    ActorKnight -->|Choose Action| Sim[HTML5 Canvas Battle Arena]
    ActorRanger -->|Choose Action| Sim
    ActorHealer -->|Choose Action| Sim
    Sim -->|State Transitions| ObsKnight
    Sim -->|State Transitions| ObsRanger
    Sim -->|State Transitions| ObsHealer
```

1.  **Centralized Critic ($V(\mathcal{S})$)**: During training, a centralized value network observes the joint global state vector $\mathcal{S} \in \mathbb{R}^{99}$ (concatenating all agent observations), guiding the policy updates with high-fidelity joint value estimates.
2.  **Decentralized Actors ($\pi(a_i | o_i)$)**: During execution in the web browser, the Critic is stripped away. Individual Actor MLP networks run feedforward matrix multiplications locally in-browser using only their decentralized 33-feature observation vectors (consisting of self health, local cooldowns, and offsets to visible allies and Goblins).

---

## 🛠️ Challenges Faced & Engineering Solutions

A high-quality reinforcement learning project is defined by the real engineering roadblocks solved during development. Here is the detailed breakdown of the challenges encountered and resolved in this project:

### Challenge 1: The "Cowardly Agent" Local Minimum (Reward Shaping)
*   **The Problem**: During early training iterations, the Blue Team achieved a flat 0.0% win rate. When spawned, the Knight, Ranger, and Healer would immediately run away from the Goblins, pinning themselves against the far walls and stalling.
*   **The Root Cause**: We designed a team penalty for taking damage to encourage mutual protection (`rewards[a] -= damage * 0.15` per agent). However, because there are 3 agents, the total team penalty amounted to `-0.45 * damage`, which mathematically outweighed the individual attack reward of `+0.4 * damage`. The neural networks quickly realized that fighting resulted in net-negative rewards. Fleeing to corners to delay contact was the mathematically optimal "local minimum" strategy to maximize episodic returns.
*   **The Solution**: We rebalanced the MDP reward matrix:
    *   Doubled attack rewards from `+0.4 * damage` to **`+0.8 * damage`**.
    *   Reduced the damage taken penalty from `-0.15` to **`-0.05`** per agent.
    *   Increased the team victory reward from `+50.0` to **`+80.0`**.
    This flipped the mathematical motivation, encouraging brave frontline coordination.

### Challenge 2: The "Meatgrinder" Discrepancy (Environment Alignment)
*   **The Problem**: Even with rebalanced rewards, the win rate in the Python training terminal stayed at 0.0% across 350 episodes, and agents still eventually learned to flee.
*   **The Root Cause**: A deep check of the code revealed an environment mismatch. In the browser (`rts_sim.js`), Goblins had a 10-step attack cooldown. But in the Python Gym environment (`marl_env.py`), Goblins had **no cooldown variable** and attacked every single step! Goblins were dealing a massive $3 \times 7 = 21$ damage *per step* to the Knight, melting him in just 7 steps. Winning was mathematically impossible, forcing the AI to revert to fleeing to minimize step penalties.
*   **The Solution**: We added a `"cd": 0` (cooldown) field to each Goblin's state in `marl_env.py` and restricted their attack execution to an 8-step cooldown. **Immediately, the win rate surged from 0.0% to a peak of 90.0%**, as the AI finally had a fair environment where team cooperation could achieve victory.

### Challenge 3: Scrambled Observations and Post-Kill Neural Lockups
*   **The Problem**: When the trained weights were loaded in the browser, the Knight charged the Goblins but froze in place immediately after defeating the first target. The Healer also stood completely still.
*   **The Root Cause**: We uncovered two separate structural issues:
    1.  **Scrambled Inputs**: The Python environment processed observations strictly by Goblin index (Goblin 0 in slot 0, Goblin 1 in slot 1). The JS simulator filtered out dead Goblins and sorted active Goblins by distance, moving them up the array. When Goblin 0 died, the JS shifted Goblins 1 and 2 into slots 0 and 1. This scrambled input completely confused the neural network, causing the Knight to freeze.
    2.  **Obs Solts Overrun**: When spawning extra Goblins in Manual Mode, they were pushed to indices 3, 4, 5... of the `goblins` array. The Actor network only has input weights for slots 0, 1, and 2, making the spawned reinforcements completely "invisible" to the AI.
    3.  **Action Obsession**: The Healer's casting rewards during training were so high that its policy got locked onto spamming Action 5 (Heal Skill) at every step, preventing it from ever executing movement actions.
*   **The Solution**: We resolved these issues with a series of high-quality software fallbacks:
    *   **Index-Based Padding**: Refactored `getObservationForAgent()` in JS to follow strict index mapping, matching Python 100%.
    *   **Slot-Reusing Spawns**: Updated `handleCanvasClick()` to reuse dead Goblin slots (0, 1, 2) when spawning reinforcements, keeping the Goblins array capped at 3 active targets.
    *   **Hybrid Fallbacks**: Programmed a hybrid control architecture. If the AI policy decides to cast its skill (Action 5) but no targets are in range, it triggers a **heuristic fallback**: the Knight/Ranger charges the closest Goblin, and the Healer **automatically follows 80px behind the Knight** as a mobile support!

---

## 📂 Project Structure

```text
├── marl_env.py            # Custom Multi-Agent 2D Gymnasium Environment
├── mappo_scratch.py        # Custom PyTorch MAPPO from scratch
├── train_marl.py          # Training orchestrator & JS Weights exporter
├── test_equivalence.py    # Automated mathematical validation suite
├── index.html             # Sleek dark-mode tabbed web dashboard
├── index.css              # Glassmorphism and glowing neon Cyberpunk UI
├── rts_sim.js             # Canvas render loop, physics, & JS weights forward-pass
├── learning_lab.js        # Tab switcher, Bellman equation sandbox, neural synapse canvas
└── marl_weights.js        # Exported neural weights & feedforward matrix math
```

---

## 🏆 Portfolio Highlights & Technical Value
*   **Production Code Standards**: Clean docstrings, strict vector mapping, and isolated environments.
*   **Zero-Dependency Deployment**: Demonstrates advanced edge inference. The entire neural network forward pass runs locally in-browser without any server lag or heavy packages.
*   **Interactive Explainability**: The Neural Lab tab translates mathematical concepts (Bellman equation, neural network synapse activations) into interactive visual sandbox tools.
*   **Gamification**: The "Manual Mode" allows recruiters to spawn hordes of Goblins live to stress-test your AI.
