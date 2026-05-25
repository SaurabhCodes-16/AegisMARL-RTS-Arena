import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from marl_env import MARLRtsEnv
from mappo_scratch import MAPPOAgent, MAPPOBuffer


def export_weights_to_js(agent, filepath="marl_weights.js"):
    """
    Exports the PyTorch Actor weights for Knight, Ranger, and Healer
    into a clean JavaScript file containing raw nested array variables
    for feedforward matrix multiplication in-browser.
    """
    weights_dict = {}

    for a in ["knight", "ranger", "healer"]:
        actor_net = agent.actors[a].network
        
        # Extract weights and biases from PyTorch layers
        # Layer 0 is Linear, 2 is Linear, 4 is Linear
        w1 = actor_net[0].weight.data.cpu().numpy().tolist()  # Shape: (64, 33)
        b1 = actor_net[0].bias.data.cpu().numpy().tolist()    # Shape: (64,)
        w2 = actor_net[2].weight.data.cpu().numpy().tolist()  # Shape: (64, 64)
        b2 = actor_net[2].bias.data.cpu().numpy().tolist()    # Shape: (64,)
        w3 = actor_net[4].weight.data.cpu().numpy().tolist()  # Shape: (6, 64)
        b3 = actor_net[4].bias.data.cpu().numpy().tolist()    # Shape: (6,)

        weights_dict[a] = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "w3": w3,
            "b3": b3
        }

    # Format into a JS file
    js_content = f"// Automatically generated MARL neural network weights\n"
    js_content += f"// Model type: MAPPO actor weights (MLP: 33 -> 64 -> 64 -> 6)\n\n"
    js_content += f"const marl_weights = {json.dumps(weights_dict, indent=2)};\n\n"
    
    # Add JS forward pass math engine so the web app requires no external libraries
    js_content += """
// Feedforward matrix-math inference engine in Javascript
function predictAction(observation, agentType) {
    if (!marl_weights[agentType]) {
        console.error("Unknown agent type: " + agentType);
        return 0; // Return Idle
    }

    const w = marl_weights[agentType];
    
    // Check if dead (obs is all zeros)
    let isDead = true;
    for (let i = 0; i < observation.length; i++) {
        if (observation[i] !== 0) {
            isDead = false;
            break;
        }
    }
    if (isDead) return 0; // Idle

    // Layer 1: Linear + Tanh
    let h1 = [];
    for (let i = 0; i < w.w1.length; i++) {
        let sum = w.b1[i];
        for (let j = 0; j < observation.length; j++) {
            sum += w.w1[i][j] * observation[j];
        }
        h1.push(Math.tanh(sum));
    }

    // Layer 2: Linear + Tanh
    let h2 = [];
    for (let i = 0; i < w.w2.length; i++) {
        let sum = w.b2[i];
        for (let j = 0; j < h1.length; j++) {
            sum += w.w2[i][j] * h1[j];
        }
        h2.push(Math.tanh(sum));
    }

    // Layer 3: Linear (Action logits)
    let logits = [];
    for (let i = 0; i < w.w3.length; i++) {
        let sum = w.b3[i];
        for (let j = 0; j < h2.length; j++) {
            sum += w.w3[i][j] * h2[j];
        }
        logits.push(sum);
    }

    // Softmax / Argmax for Action Choice
    // In simulation, we can choose the argmax (deterministic) for optimal showcase
    let maxVal = -Infinity;
    let bestAction = 0;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxVal) {
            maxVal = logits[i];
            bestAction = i;
        }
    }
    return bestAction;
}
"""

    with open(filepath, "w") as f:
        f.write(js_content)
    print(f"Neural network weights successfully exported to browser asset: {filepath}")


def train():
    env = MARLRtsEnv()
    
    # Hyperparameters
    num_episodes = 500
    buffer_capacity = 2000
    
    # Initialize Agent & Buffer
    agent = MAPPOAgent(
        lr_actor=4e-4, 
        lr_critic=1.5e-3, 
        gamma=0.98, 
        gae_lambda=0.95, 
        clip_ratio=0.2
    )
    buffer = MAPPOBuffer(num_agents=3, obs_dim=33, state_dim=99, size=buffer_capacity)

    # Tracking metrics
    episodic_rewards = []
    win_rates = []
    goblin_kills = []
    avg_losses = []
    
    win_history = []  # Last 30 episodes for moving average win rate

    print("=" * 60)
    print("      Starting Custom PyTorch MAPPO RTS Arena Training        ")
    print("=" * 60)

    for ep in range(1, num_episodes + 1):
        obs_dict, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Construct joint state (for Centralized Critic) by concatenating all 3 obs
            joint_state = np.concatenate([obs_dict[a] for a in ["knight", "ranger", "healer"]])
            
            # Select actions
            actions, log_probs = agent.select_actions(obs_dict, deterministic=False)
            
            # Get Critic Value
            value = agent.get_value(joint_state)
            
            # Step environment
            next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
            
            # Cooperative shared team reward: average reward of agents to foster teamwork
            team_reward = sum(rewards.values()) / 3.0
            episode_reward += team_reward
            
            done = terminated or truncated
            step_count += 1
            
            # Store transition in PPO rollout buffer
            buffer.store(
                obs_dict=obs_dict,
                state=joint_state,
                actions_dict=actions,
                log_probs_dict=log_probs,
                reward=team_reward,
                value=value,
                done=float(terminated)  # Don't bootstrap if terminated naturally
            )
            
            obs_dict = next_obs_dict
            
            # Buffer capacity full: update PPO policies early
            if buffer.ptr >= buffer.size:
                losses = agent.update(buffer)
                avg_losses.append(losses)

        # Episode concluded
        episodic_rewards.append(episode_reward)
        gobs_slain = 3 - info["goblins_alive"]
        goblin_kills.append(gobs_slain)
        
        # Did the team win? (All Goblins dead, at least one Blue agent alive)
        won = info["goblins_alive"] == 0 and info["blue_alive_count"] > 0
        win_history.append(1.0 if won else 0.0)
        if len(win_history) > 30:
            win_history.pop(0)
        win_rates.append(np.mean(win_history))

        # Perform PPO update at the end of each episode if buffer has collected enough transitions
        if buffer.ptr >= 150:  # Train after accumulating some transitions
            losses = agent.update(buffer)
            avg_losses.append(losses)

        # Periodic logging
        if ep % 25 == 0 or ep == 1:
            avg_rew = np.mean(episodic_rewards[-25:]) if ep > 1 else episode_reward
            print(f"Episode {ep:3d} | Win Rate: {win_rates[-1]*100:5.1f}% | Avg Reward: {avg_rew:7.2f} | Slain: {gobs_slain}/3 | Steps: {step_count}")

    print("\nTraining Complete! Saving models and generating assets...")

    # Save pytorch weights
    agent.save_weights("mappo_rts")
    print("PyTorch model checkpoints saved as 'mappo_rts_actor_<class>.pth' and 'mappo_rts_critic.pth'.")

    # Export JS file for web integration
    export_weights_to_js(agent, "marl_weights.js")

    # Plot metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episodic_rewards, color="#00ffcc", alpha=0.3, label="Raw Return")
    # Moving average
    ma_rewards = np.convolve(episodic_rewards, np.ones(20)/20, mode="valid")
    plt.plot(np.arange(19, len(episodic_rewards)), ma_rewards, color="#3b82f6", linewidth=2, label="Moving Average")
    plt.title("MAPPO Episodic Joint Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, color="#dddddd", linestyle="--")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(win_rates, color="#ff007f", linewidth=2)
    plt.title("Team Win Rate Over Time (30-Ep MA)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, color="#dddddd", linestyle="--")

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=200)
    print("Performance graph exported to disk: learning_curves.png")


if __name__ == "__main__":
    train()
