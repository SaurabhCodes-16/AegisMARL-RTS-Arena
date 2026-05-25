import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):
    """
    Decentralized Actor Network.
    Inputs: Local observation vector (33 floats)
    Outputs: Action logits (6 discrete actions)
    """
    def __init__(self, obs_dim=33, act_dim=6):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs):
        return self.network(obs)

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized Critic Network (CTDE).
    Inputs: Concatenated global state vector (33 * 3 = 99 floats)
    Outputs: Team State Value V(S) (1 scalar float)
    """
    def __init__(self, state_dim=99):
        super(CentralizedCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)


class MAPPOBuffer:
    """
    Rollout Buffer for storing trajectories of multiple agents.
    """
    def __init__(self, num_agents=3, obs_dim=33, state_dim=99, size=1500):
        self.size = size
        self.num_agents = num_agents

        self.obs = {a: np.zeros((size, obs_dim), dtype=np.float32) for a in ["knight", "ranger", "healer"]}
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = {a: np.zeros((size,), dtype=np.int64) for a in ["knight", "ranger", "healer"]}
        self.log_probs = {a: np.zeros((size,), dtype=np.float32) for a in ["knight", "ranger", "healer"]}
        self.rewards = np.zeros((size,), dtype=np.float32)  # Shared joint team reward for cooperation
        self.values = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)

        self.ptr = 0

    def store(self, obs_dict, state, actions_dict, log_probs_dict, reward, value, done):
        if self.ptr >= self.size:
            return False  # Buffer full

        for a in ["knight", "ranger", "healer"]:
            self.obs[a][self.ptr] = obs_dict[a]
            self.actions[a][self.ptr] = actions_dict[a]
            self.log_probs[a][self.ptr] = log_probs_dict[a]

        self.states[self.ptr] = state
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        self.ptr += 1
        return True

    def clear(self):
        self.ptr = 0


class MAPPOAgent:
    """
    MAPPO Core Controller coordinating separate Actors and a single Centralized Critic.
    """
    def __init__(self, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, c1=0.5, c2=0.01):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.c1 = c1  # Critic loss coefficient
        self.c2 = c2  # Entropy regularization coefficient

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Three Actors (Decentralized policies)
        self.actors = {
            "knight": Actor(obs_dim=33, act_dim=6).to(self.device),
            "ranger": Actor(obs_dim=33, act_dim=6).to(self.device),
            "healer": Actor(obs_dim=33, act_dim=6).to(self.device)
        }

        # Centralized Critic
        self.critic = CentralizedCritic(state_dim=99).to(self.device)

        # Optimizers
        self.actor_optimizers = {
            a: optim.Adam(self.actors[a].parameters(), lr=lr_actor) for a in ["knight", "ranger", "healer"]
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(self, obs_dict, deterministic=False):
        """
        Inference: select actions for each agent using local observations.
        """
        actions = {}
        log_probs = {}

        for a in ["knight", "ranger", "healer"]:
            obs_tensor = torch.FloatTensor(obs_dict[a]).to(self.device)
            # Handle dead agents (zeros obs)
            if torch.all(obs_tensor == 0):
                actions[a] = 0  # Idle
                log_probs[a] = 0.0
                continue

            with torch.no_grad():
                act, log_p, _ = self.actors[a].get_action(obs_tensor, deterministic)
            actions[a] = act.item()
            log_probs[a] = log_p.item()

        return actions, log_probs

    def get_value(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            val = self.critic(state_tensor)
        return val.item()

    def update(self, buffer):
        """
        MAPPO update using collected rollout transitions.
        """
        # Convert buffer data to torch tensors
        obs_tensors = {
            a: torch.FloatTensor(buffer.obs[a][:buffer.ptr]).to(self.device)
            for a in ["knight", "ranger", "healer"]
        }
        states = torch.FloatTensor(buffer.states[:buffer.ptr]).to(self.device)
        actions = {
            a: torch.LongTensor(buffer.actions[a][:buffer.ptr]).to(self.device)
            for a in ["knight", "ranger", "healer"]
        }
        old_log_probs = {
            a: torch.FloatTensor(buffer.log_probs[a][:buffer.ptr]).to(self.device)
            for a in ["knight", "ranger", "healer"]
        }

        rewards = buffer.rewards[:buffer.ptr]
        values = buffer.values[:buffer.ptr]
        dones = buffer.dones[:buffer.ptr]

        # --- Compute GAE (Generalized Advantage Estimation) ---
        advantages = np.zeros(buffer.ptr, dtype=np.float32)
        returns = np.zeros(buffer.ptr, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(buffer.ptr - 1)):
            next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        # Final step boundary handling
        returns[-1] = rewards[-1] + self.gamma * (1.0 - dones[-1]) * values[-1]
        advantages[-1] = returns[-1] - values[-1]

        # Convert to torch tensors
        adv_tensor = torch.FloatTensor(advantages).to(self.device)
        # Normalize advantages for stability
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Optimize for multiple epochs
        ppo_epochs = 5
        batch_size = 64

        actor_losses_log = {a: 0.0 for a in ["knight", "ranger", "healer"]}
        critic_loss_log = 0.0

        for epoch in range(ppo_epochs):
            indices = np.arange(buffer.ptr)
            np.random.shuffle(indices)

            for start in range(0, buffer.ptr, batch_size):
                end = min(start + batch_size, buffer.ptr)
                batch_indices = indices[start:end]

                b_states = states[batch_indices]
                b_advs = adv_tensor[batch_indices]
                b_returns = returns_tensor[batch_indices]

                # --- 1. Update Decentralized Actors ---
                for a in ["knight", "ranger", "healer"]:
                    b_obs = obs_tensors[a][batch_indices]
                    b_acts = actions[a][batch_indices]
                    b_old_log_probs = old_log_probs[a][batch_indices]

                    # Filter out indices where the agent is dead (obs is all zeros)
                    alive_mask = torch.sum(torch.abs(b_obs), dim=-1) > 0

                    if not torch.any(alive_mask):
                        continue

                    # Forward pass
                    logits = self.actors[a](b_obs[alive_mask])
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(b_acts[alive_mask])
                    entropy = dist.entropy().mean()

                    # Ratio
                    ratios = torch.exp(new_log_probs - b_old_log_probs[alive_mask])

                    # PPO Clipped Loss
                    surr1 = ratios * b_advs[alive_mask]
                    surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advs[alive_mask]
                    actor_loss = -torch.min(surr1, surr2).mean() - self.c2 * entropy

                    # Optimize Actor
                    self.actor_optimizers[a].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[a].parameters(), 1.0)
                    self.actor_optimizers[a].step()

                    actor_losses_log[a] += actor_loss.item()

                # --- 2. Update Centralized Critic ---
                val_predictions = self.critic(b_states).squeeze(-1)
                critic_loss = self.c1 * nn.MSELoss()(val_predictions, b_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()

                critic_loss_log += critic_loss.item()

        # Clear buffer after training epoch
        buffer.clear()

        # Return average losses for reporting
        num_steps = max(1, (buffer.ptr // batch_size) * ppo_epochs)
        return {
            "actor_knight_loss": actor_losses_log["knight"] / num_steps,
            "actor_ranger_loss": actor_losses_log["ranger"] / num_steps,
            "actor_healer_loss": actor_losses_log["healer"] / num_steps,
            "critic_loss": critic_loss_log / num_steps
        }

    def save_weights(self, path_prefix):
        """
        Saves actors and critic network states.
        """
        for a in ["knight", "ranger", "healer"]:
            torch.save(self.actors[a].state_dict(), f"{path_prefix}_actor_{a}.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

    def load_weights(self, path_prefix):
        """
        Loads actors and critic network states.
        """
        for a in ["knight", "ranger", "healer"]:
            self.actors[a].load_state_dict(torch.load(f"{path_prefix}_actor_{a}.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth", map_location=self.device))
