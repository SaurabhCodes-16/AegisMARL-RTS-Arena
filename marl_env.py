import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MARLRtsEnv(gym.Env):
    """
    Custom 2D RTS Battle Arena Multi-Agent Environment.
    Blue Team (Controlled by RL):
      - Knight (melee, high HP)
      - Ranger (ranged, low HP)
      - Healer (healer, low HP)
    Red Team (Heuristic Goblins):
      - 3 aggressive Goblins that attack the closest Blue unit.

    Coordinates: 100x100 space.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(MARLRtsEnv, self).__init__()

        self.agents = ["knight", "ranger", "healer"]

        # Action Space:
        # 0: Idle
        # 1: Move North (y increase)
        # 2: Move South (y decrease)
        # 3: Move East (x increase)
        # 4: Move West (x decrease)
        # 5: Execute Class Skill (Knight: Melee Attack, Ranger: Ranged Attack, Healer: Heal Ally)
        self.action_space = spaces.Discrete(6)

        # Observation Space per Agent: Size 33
        # Self (7): [HP%, Type_Knight, Type_Ranger, Type_Healer, Cooldown%, x, y]
        # Allies (2 * 7 = 14): [Active, RelX, RelY, HP%, Type_Knight, Type_Ranger, Type_Healer] x 2
        # Enemies (3 * 4 = 12): [Active, RelX, RelY, HP%] x 3
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(33,), dtype=np.float32
        )

        # Game parameters
        self.max_steps = 150
        self.current_step = 0

        # Unit base stats
        self.stats = {
            "knight": {"max_hp": 150.0, "dmg": 18.0, "range": 12.0, "cooldown": 8, "speed": 2.0},
            "ranger": {"max_hp": 80.0, "dmg": 10.0, "range": 45.0, "cooldown": 10, "speed": 2.5},
            "healer": {"max_hp": 60.0, "heal": 12.0, "range": 35.0, "cooldown": 12, "speed": 2.2},
            "goblin": {"max_hp": 75.0, "dmg": 7.0, "range": 8.0, "speed": 1.6}
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Blue team initial states
        self.unit_states = {
            "knight": {"hp": self.stats["knight"]["max_hp"], "pos": np.array([20.0, 35.0]), "cd": 0, "active": 1},
            "ranger": {"hp": self.stats["ranger"]["max_hp"], "pos": np.array([15.0, 50.0]), "cd": 0, "active": 1},
            "healer": {"hp": self.stats["healer"]["max_hp"], "pos": np.array([20.0, 65.0]), "cd": 0, "active": 1}
        }

        # Red team (Goblins)
        self.goblin_states = [
            {"hp": self.stats["goblin"]["max_hp"], "pos": np.array([80.0, 30.0]), "active": 1, "cd": 0},
            {"hp": self.stats["goblin"]["max_hp"], "pos": np.array([85.0, 50.0]), "active": 1, "cd": 0},
            {"hp": self.stats["goblin"]["max_hp"], "pos": np.array([80.0, 70.0]), "active": 1, "cd": 0}
        ]

        # Return dict of observations
        return self._get_observations(), {}

    def _get_observations(self):
        obs_dict = {}

        for agent in self.agents:
            agent_state = self.unit_states[agent]
            if not agent_state["active"]:
                # Dead agents receive zeros
                obs_dict[agent] = np.zeros((33,), dtype=np.float32)
                continue

            # 1. Self Features (7)
            hp_pct = agent_state["hp"] / self.stats[agent]["max_hp"]
            cd_pct = agent_state["cd"] / self.stats[agent]["cooldown"]
            pos = agent_state["pos"] / 100.0  # Normalize to [0, 1]
            one_hot_type = [
                1.0 if agent == "knight" else 0.0,
                1.0 if agent == "ranger" else 0.0,
                1.0 if agent == "healer" else 0.0
            ]
            self_feat = [hp_pct] + one_hot_type + [cd_pct] + list(pos)

            # 2. Allies Features (14) - 2 allies
            allies_feat = []
            allies = [a for a in self.agents if a != agent]
            for ally in allies:
                ally_state = self.unit_states[ally]
                if ally_state["active"]:
                    rel_pos = (ally_state["pos"] - agent_state["pos"]) / 100.0  # Normalized relative pos
                    ally_hp_pct = ally_state["hp"] / self.stats[ally]["max_hp"]
                    ally_one_hot = [
                        1.0 if ally == "knight" else 0.0,
                        1.0 if ally == "ranger" else 0.0,
                        1.0 if ally == "healer" else 0.0
                    ]
                    allies_feat += [1.0] + list(rel_pos) + [ally_hp_pct] + ally_one_hot
                else:
                    allies_feat += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # 3. Enemies Features (12) - 3 Goblins
            enemies_feat = []
            for gob in self.goblin_states:
                if gob["active"]:
                    rel_pos = (gob["pos"] - agent_state["pos"]) / 100.0
                    gob_hp_pct = gob["hp"] / self.stats["goblin"]["max_hp"]
                    enemies_feat += [1.0] + list(rel_pos) + [gob_hp_pct]
                else:
                    enemies_feat += [0.0, 0.0, 0.0, 0.0]

            obs = np.array(self_feat + allies_feat + enemies_feat, dtype=np.float32)
            obs_dict[agent] = obs

        return obs_dict

    def step(self, actions):
        """
        Executes actions for Blue team, steps Goblim AI, and computes rewards/states.
        'actions' is a dict: {'knight': action_int, 'ranger': action_int, 'healer': action_int}
        """
        self.current_step += 1

        rewards = {a: 0.0 for a in self.agents}
        action_log = {a: "Idle" for a in self.agents}

        # --- 1. Update Cooldowns ---
        for agent in self.agents:
            if self.unit_states[agent]["cd"] > 0:
                self.unit_states[agent]["cd"] -= 1

        # --- 2. Move Controlled Agents ---
        for agent in self.agents:
            state = self.unit_states[agent]
            if not state["active"]:
                continue

            act = actions.get(agent, 0)
            speed = self.stats[agent]["speed"]

            if act == 1:  # North
                state["pos"][1] = min(100.0, state["pos"][1] + speed)
                action_log[agent] = "Move North"
            elif act == 2:  # South
                state["pos"][1] = max(0.0, state["pos"][1] - speed)
                action_log[agent] = "Move South"
            elif act == 3:  # East
                state["pos"][0] = min(100.0, state["pos"][0] + speed)
                action_log[agent] = "Move East"
            elif act == 4:  # West
                state["pos"][0] = max(0.0, state["pos"][0] - speed)
                action_log[agent] = "Move West"

        # --- 3. Execute Class Skills ---
        for agent in self.agents:
            state = self.unit_states[agent]
            if not state["active"]:
                continue

            act = actions.get(agent, 0)
            if act == 5:  # Execute class-specific primary skill
                if state["cd"] == 0:
                    action_log[agent] = "Skill activated!"
                    state["cd"] = self.stats[agent]["cooldown"]

                    # 3A. KNIGHT & RANGER: Attack nearest alive Goblin
                    if agent in ["knight", "ranger"]:
                        # Find valid enemies in range
                        enemies_in_range = []
                        for idx, gob in enumerate(self.goblin_states):
                            if gob["active"]:
                                dist = np.linalg.norm(gob["pos"] - state["pos"])
                                if dist <= self.stats[agent]["range"]:
                                    enemies_in_range.append((idx, dist, gob))

                        if enemies_in_range:
                            # Attack the nearest one
                            enemies_in_range.sort(key=lambda x: x[1])
                            target_idx, _, target_gob = enemies_in_range[0]

                            dmg = self.stats[agent]["dmg"]
                            target_gob["hp"] -= dmg
                            rewards[agent] += dmg * 0.8  # Reward for dealing damage (Increased)
                            action_log[agent] = f"Attacked Goblin {target_idx}"

                            # Check death
                            if target_gob["hp"] <= 0:
                                target_gob["active"] = 0
                                target_gob["hp"] = 0
                                rewards[agent] += 20.0  # Killing blow bonus (Increased)
                                action_log[agent] = f"Defeated Goblin {target_idx}!"

                    # 3B. HEALER: Heal lowest HP alive ally
                    elif agent == "healer":
                        allies_in_range = []
                        for ally in self.agents:
                            ally_state = self.unit_states[ally]
                            if ally_state["active"] and ally_state["hp"] < self.stats[ally]["max_hp"]:
                                dist = np.linalg.norm(ally_state["pos"] - state["pos"])
                                if dist <= self.stats["healer"]["range"]:
                                    allies_in_range.append((ally, ally_state))

                        if allies_in_range:
                            # Heal the ally with the lowest absolute HP percentage
                            allies_in_range.sort(key=lambda x: x[1]["hp"] / self.stats[x[0]]["max_hp"])
                            target_ally, target_state = allies_in_range[0]

                            heal = self.stats["healer"]["heal"]
                            prev_hp = target_state["hp"]
                            target_state["hp"] = min(self.stats[target_ally]["max_hp"], target_state["hp"] + heal)
                            actual_healed = target_state["hp"] - prev_hp

                            # Reward healer
                            rewards["healer"] += actual_healed * 1.2  # Increased
                            if prev_hp / self.stats[target_ally]["max_hp"] < 0.35:
                                rewards["healer"] += 10.0  # Save-life critical heal bonus (Increased)
                            action_log[agent] = f"Healed {target_ally}"
                        else:
                            # No one needs healing, reset CD slightly to not punish
                            state["cd"] = 2
                else:
                    action_log[agent] = "Skill on cooldown"

        # --- 4. Goblin (Enemy AI) Step ---
        for idx, gob in enumerate(self.goblin_states):
            if not gob["active"]:
                continue

            # Decrement Goblin attack cooldown
            if gob["cd"] > 0:
                gob["cd"] -= 1

            # Find closest alive Blue agent
            alive_agents = [a for a in self.agents if self.unit_states[a]["active"]]
            if not alive_agents:
                continue

            dists = [np.linalg.norm(self.unit_states[a]["pos"] - gob["pos"]) for a in alive_agents]
            closest_idx = np.argmin(dists)
            target_agent = alive_agents[closest_idx]
            target_state = self.unit_states[target_agent]
            target_dist = dists[closest_idx]

            # If in range, attack
            if target_dist <= self.stats["goblin"]["range"]:
                if gob["cd"] == 0:
                    gob["cd"] = 8  # Cooldown of 8 steps (0.8s) to balance the battle
                    dmg = self.stats["goblin"]["dmg"]
                    target_state["hp"] -= dmg

                    # Heavily penalize the team if their agents take damage
                    for a in self.agents:
                        rewards[a] -= dmg * 0.05  # Reduced penalty (from 0.15)

                    # Check ally death
                    if target_state["hp"] <= 0:
                        target_state["active"] = 0
                        target_state["hp"] = 0
                        # Team penalty for death
                        for a in self.agents:
                            rewards[a] -= 20.0
            else:
                # Move towards target
                dir_vec = target_state["pos"] - gob["pos"]
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                gob["pos"] += dir_vec * self.stats["goblin"]["speed"]

        # --- 5. Step Penalty and Terminal Checks ---
        # Add a tiny step penalty to encourage fast clears
        for a in self.agents:
            rewards[a] -= 0.1

        # Check termination conditions
        blue_alive = any(self.unit_states[a]["active"] for a in self.agents)
        red_alive = any(gob["active"] for gob in self.goblin_states)

        terminated = False
        truncated = self.current_step >= self.max_steps

        # Victory!
        if not red_alive:
            terminated = True
            for a in self.agents:
                rewards[a] += 80.0  # Massive team win bonus (Increased)

        # Defeat!
        elif not blue_alive:
            terminated = True
            for a in self.agents:
                rewards[a] -= 30.0  # Defeat penalty

        # Pack info
        info = {
            "step": self.current_step,
            "action_log": action_log,
            "knight_hp": self.unit_states["knight"]["hp"],
            "ranger_hp": self.unit_states["ranger"]["hp"],
            "healer_hp": self.unit_states["healer"]["hp"],
            "goblins_alive": sum(1 for gob in self.goblin_states if gob["active"]),
            "blue_alive_count": sum(1 for a in self.agents if self.unit_states[a]["active"])
        }

        return self._get_observations(), rewards, terminated, truncated, info
