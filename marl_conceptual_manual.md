# AegisMARL: Comprehensive Conceptual & Process Manual

This manual provides an in-depth, rigorous mathematical and system-level explanation of the **AegisMARL Cooperative RTS Arena** codebase. It covers core Multi-Agent Reinforcement Learning (MARL) theory, the Proximal Policy Optimization (PPO) training pipeline, the Edge-Inference compiler logic, and detailed diagnostic breakdowns of the engineering challenges resolved.

---

## Table of Contents
1. Core Paradigms of Multi-Agent Reinforcement Learning (MARL)
2. Centralized Training with Decentralized Execution (CTDE)
3. Mathematical Foundations of MAPPO and GAE
4. Markov Decision Process (MDP) System Design
5. In-Browser Edge Inference Math
6. Case Studies: Root-Cause Debugging & Resolutions

---

## 1. Core Paradigms of Multi-Agent Reinforcement Learning (MARL)

### 1.1 Single-Agent vs. Multi-Agent Environments
In a standard single-agent Markov Decision Process (MDP), the environment is assumed to be **stationary**. The transition probability function $\mathcal{P}(s' | s, a)$ depends exclusively on the current state $s$ and the single action $a$ of the agent.

In a Multi-Agent environment, stationarity is broken. Multiple agents (Knight, Ranger, Healer) learn and update their policies $\pi\_i$ simultaneously. From the individual perspective of the Knight, the environment transition dynamics $\mathcal{P}(s' | s, a\_{\text{knight}}, \mathbf{a}\_{-\text{knight}})$ shift dynamically as the Ranger and Healer alter their policies. This is known as the **Non-Stationarity Problem**. If agents are trained independently using standard single-agent algorithms, their policies will fail to converge because the mathematical foundations of their value estimators are shifting constantly.

---

## 2. Centralized Training with Decentralized Execution (CTDE)

CTDE is the industry-standard paradigm designed to stabilize learning under non-stationarity.

```
+-------------------------------------------------------------------------+
|                         TRAINING PHASE (Centralized)                    |
|                                                                         |
|   Knight Obs (33) \                                                     |
|   Ranger Obs (33)  ==> Concatenated State S (99) ==> Central Critic V(S) |
|   Healer Obs (33) /                                                     |
+------------------------------------+------------------------------------+
                                     | Generates State Values V(S)
                                     v
+------------------------------------+------------------------------------+
|                         EXECUTION PHASE (Decentralized)                 |
|                                                                         |
|   Knight Obs (33) =======> Actor Policy (Knight MLP) =======> Action    |
|   Ranger Obs (33) =======> Actor Policy (Ranger MLP) =======> Action    |
|   Healer Obs (33) =======> Actor Policy (Healer MLP) =======> Action    |
+-------------------------------------------------------------------------+
```

1.  **Centralized Training**: During the offline training phase in PyTorch, we have access to the global simulator state. We concatenate all agents' individual observations into a single joint vector $\mathcal{S} \in \mathbb{R}^{99}$. A **Centralized Critic** network evaluates this joint vector to estimate the global team state value $V(\mathcal{S})$. Because the critic sees what *everyone* is doing, it can accurately evaluate the state transition probabilities, neutralizing the non-stationarity problem.
2.  **Decentralized Execution**: During runtime (in the browser), the Centralized Critic is stripped away. The agents operate independently as decentralized actors. The Knight only feeds its local 33-feature observation $o\_{\text{knight}}$ into its Actor MLP to get its action. There is no real-time communication channel or global coordinate sync; coordination emerges purely from their decentralized policy parameters.

---

## 3. Mathematical Foundations of MAPPO and GAE

### 3.1 MAPPO Policy Optimization
AegisMARL implements **Multi-Agent Proximal Policy Optimization (MAPPO)**. PPO belongs to the family of actor-critic policy gradient methods.

To prevent destructive policy updates (where a bad gradient step completely destroys the network's capabilities), PPO uses a **Clipped Surrogate Objective** that constrains the policy update step within a trust region.

For each agent $i$, we define the probability ratio $r\_t(\theta\_i)$ between the new policy and the old policy:
$$r\_t(\theta\_i) = \frac{\pi\_{\theta\_i}(a\_{t,i} | o\_{t,i})}{\pi\_{\theta\_{\text{old}, i}}(a\_{t,i} | o\_{t,i})}$$

The clipped loss objective is defined as:
$$L^{\text{CLIP}}(\theta\_i) = \hat{\mathbb{E}}\_t \left[ \min\left(r\_t(\theta\_i) \hat{A}\_t, \, \text{clip}(r\_t(\theta\_i), 1-\epsilon, 1+\epsilon) \hat{A}\_t\right) \right]$$

*   If the advantage $\hat{A}\_t$ is positive, it means the chosen action was better than average. The loss function encourages the policy to increase the probability of this action, but the `clip` bounds the maximum increase to $1+\epsilon$ to prevent over-adjustment.
*   If the advantage $\hat{A}\_t$ is negative, it means the action was worse than average, and the loss function discourages it.

### 3.2 Generalized Advantage Estimation (GAE)
The advantage $\hat{A}\_t$ measures how much better a chosen action is compared to the expected baseline value of the state. We calculate advantages using GAE, which introduces a parameter $\lambda$ to balance variance (high in raw returns) and bias (high in single-step value estimates).

The temporal difference error $\delta\_t^V$ at step $t$ is computed using the centralized critic $V\_\phi(\mathcal{S})$:
$$\delta\_t^V = r\_t + \gamma V\_\phi(\mathcal{S}\_{t+1}) - V\_\phi(\mathcal{S}\_t)$$

The GAE advantage $\hat{A}\_t$ is calculated recursively as:
$$\hat{A}\_t = \sum\_{l=0}^{\infty} (\gamma \lambda)^l \delta\_{t+l}^V$$
where $\gamma$ is the discount factor (e.g., $0.98$) and $\lambda$ is the GAE parameter (e.g., $0.95$).

---

## 4. Markov Decision Process (MDP) System Design

The battle mechanics are mapped mathematically as follows:

### 4.1 State and Observation Vector ($\mathbb{R}^{33}$)
The 33-dimensional float array represents a strictly normalized ego-centric observation:
```text
Indices [0-6]   : Self Features
                  - Index 0   : Current HP % (0.0 to 1.0)
                  - Indices 1-3: One-hot Type: [Knight?, Ranger?, Healer?]
                  - Index 4   : Current Cooldown % (0.0 to 1.0)
                  - Indices 5-6: Normalized Global Coordinates [X/100, Y/100]

Indices [7-20]  : Allies Features (Nearest 2 allies, 7 features each)
                  - Index 7   : Ally Active Status (1.0 = Alive, 0.0 = Dead)
                  - Indices 8-9: Relative offsets [ (Ally_X - Self_X)/100, (Ally_Y - Self_Y)/100 ]
                  - Index 10  : Ally HP %
                  - Indices 11-13: Ally One-hot Type

Indices [21-32] : Enemies Features (3 Goblins, 4 features each, index-mapped)
                  - Index 21  : Goblin Active Status (1.0 = Alive, 0.0 = Dead)
                  - Indices 22-23: Relative offsets [ (Goblin_X - Self_X)/100, (Goblin_Y - Self_Y)/100 ]
                  - Index 24  : Goblin HP %
```

### 4.2 Action Processing
The model outputs one of 6 discrete actions. 
*   Actions `1` to `4` directly modify coordinates in the 2D plane:
    $$\Delta X, \Delta Y = \text{Cardinal\_Vector} \times \text{Speed}$$
*   Action `5` executes the class skill. The game coordinates physics ranges:
    *   **Knight Cleave**: Attacks closest active Goblin if distance $\le 12.0$ units.
    *   **Ranger Arrow**: Fires projectile at lowest-HP active Goblin if distance $\le 45.0$ units.
    *   **Healer Mend**: Restores health to lowest-HP active ally if distance $\le 35.0$ units.

---

## 5. In-Browser Edge Inference Math

To deploy a zero-dependency web dashboard, AegisMARL exports the trained actor networks from PyTorch as plain JSON arrays inside `marl_weights.js`. 

Each actor is a Multi-Layer Perceptron (MLP) with the architecture: **33 Inputs $\rightarrow$ 64 Hidden Units $\rightarrow$ 64 Hidden Units $\rightarrow$ 6 Outputs**.

### 5.1 Feedforward Layer Computations
The browser performs matrix multiplication natively in JavaScript using these mathematical layers:

#### Layer 1 (Hidden Layer 1):
$$\mathbf{h}\_1 = \tanh(\mathbf{W}\_1 \cdot \mathbf{x} + \mathbf{b}\_1)$$
where:
*   $\mathbf{x}$ is the 33-dimensional observation vector.
*   $\mathbf{W}\_1$ is the weight matrix of shape $64 \times 33$.
*   $\mathbf{b}\_1$ is the bias vector of shape $64$.
*   $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ is the hyperbolic tangent activation function, mapping values to $[-1, 1]$.

#### Layer 2 (Hidden Layer 2):
$$\mathbf{h}\_2 = \tanh(\mathbf{W}\_2 \cdot \mathbf{h}\_1 + \mathbf{b}\_2)$$
where $\mathbf{W}\_2$ has shape $64 \times 64$, and $\mathbf{b}\_2$ has shape $64$.

#### Layer 3 (Output Logits):
$$\mathbf{z} = \mathbf{W}\_3 \cdot \mathbf{h}\_2 + \mathbf{b}\_3$$
where $\mathbf{W}\_3$ has shape $6 \times 64$, and $\mathbf{b}\_3$ has shape $6$.

### 5.2 Action Selection (Deterministic Argmax)
In the browser evaluation environment, we select the action deterministically to showcase the absolute optimal, trained performance:
$$\text{Selected Action} = \arg\max\_{j \in [0, 5]} z\_j$$

---

## 6. Case Studies: Root-Cause Debugging & Resolutions

Below are the detailed case studies of the critical engineering bottlenecks solved during the development of this platform:

### Case Study A: The Cowardly Agent Local Minimum
*   **Symptom**: The Blue Team agents immediately ran away to the far western boundaries of the canvas and pinned themselves to the walls, resulting in a 0.0% win rate.
*   **Root Cause**: The MDP design originally incorporated a team penalty for taking damage (`rewards[a] -= dmg * 0.15` per agent). Because there are 3 agents in the cooperative team, any hit on the Knight resulted in a total team penalty of $3 \times -0.15 = -0.45 \times \text{dmg}$. However, dealing damage only returned an individual reward of `+0.4 * dmg`. The net mathematical reward for fighting was negative. The neural network successfully exploited this reward formulation, discovering that running away to corners postponed contact, maximizing returns by delaying damage.
*   **Resolution**: Rebalanced the reward shaping:
    1.  Increased attack rewards to **`+0.8 * dmg`**.
    2.  Reduced the team hit penalty to **`-0.05 * dmg`** per agent.
    3.  Increased the global team win reward from `+50.0` to **`+80.0`** and killing blows to `+20.0`.
    This shift made combat highly profitable, motivating the policy gradient to move forward.

### Case Study B: The Meatgrinder Environment Discrepancy
*   **Symptom**: Even with rebalanced rewards, the win rate in the training script remained flat at 0.0% over 350 episodes.
*   **Root Cause**: A core environment physics mismatch existed between the training script (`marl_env.py`) and the browser rendering code (`rts_sim.js`). In JS, Goblins had a 10-step attack cooldown. In the training environment, Goblins had no cooldown variable and attacked every single step! Goblins were dealing 21 damage per step to the Knight, killing him in 7 frames. Winning was physically impossible. The policy gradient defaulted back to fleeing as the only way to minimize step penalties.
*   **Resolution**: Programmed a `"cd": 0` state variable and an 8-step attack cooldown for Goblins in `marl_env.py` to match the balanced frontend. **Immediately, the win rate converged to a peak of 90.0%**, showing rapid policy optimization.

### Case Study C: State-Vector Index Scrambling & Action Overfitting
*   **Symptom**: When loaded in the browser, the Knight charged forward and defeated the first Goblin, but then froze in place. The Healer stood still and never moved from its starting position.
*   **Root Cause**:
    1.  **Observation Scrambling**: In Python, Goblins' features were mapped to indices strictly by fixed order (Goblin 0 always in slot 0, Goblin 1 in slot 1). In the browser, the JS sorted active Goblins by distance and filtered out dead ones. When Goblin 0 died, JS shifted Goblin 1 into slot 0. This index shifting scrambled the 33-feature observation array. The neural network saw coordinates of an active Goblin in a slot it expected to be zero, causing the policy to output Idle (Action 0).
    2.  **Obs Array Overrun**: When extra Goblins were spawned in Manual Mode, they were pushed to indices 3, 4, etc. The neural network only had weights for indices 0, 1, 2, making the spawned reinforcements completely "invisible" to the AI.
    3.  **Over-Reward Exploitation**: Because healing and attacking rewards were so high, the Healer locked onto spamming Action 5 (Heal Skill) and the Knight locked onto Action 5 (Attack Skill). Since they spammed Action 5 continuously, they never executed movement actions (`1-4`). Once targets moved out of range, they spammed skills in empty air.
*   **Resolution**:
    *   **Index-Based Padding**: Refactored `getObservationForAgent` in JS to strictly evaluate Goblins by fixed array indices.
    *   **Slot-Reusing Spawns**: Modified `handleCanvasClick()` to overwrite dead Goblins' slots (0, 1, 2) rather than appending new ones, keeping inputs stable.
    *   **Hybrid Fallbacks**: Added heuristic safety overrides in JS. If the policy outputs Action 5 but no targets are in range, the agent executes spatial movement toward its goal (Healer follows Knight at an 80px buffer, Knight charges the closest active Goblin).

---

## Summary of Core Code Files

1.  **`marl_env.py`**: Defends environment configurations, coordinate spaces, and steps.
2.  **`mappo_scratch.py`**: Defines the Actor-Critic networks and executes GAE / PPO mathematical clipping optimizations.
3.  **`train_marl.py`**: Executes the training loops and exports parameters into plain text JS arrays.
4.  **`test_equivalence.py`**: Asserts mathematical output equivalence between PyTorch and JavaScript.
5.  **`rts_sim.js`**: Drives the browser rendering loop and runs local feedforward matrix multiplications.
