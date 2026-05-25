import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        # State: cars waiting in [North, South, East, West]
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(4,),
            dtype=np.float32
        )

        # Actions:
        # 0 = North-South green
        # 1 = East-West green
        self.action_space = spaces.Discrete(2)

        self.state = None
        self.max_steps = 200
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.random.randint(0, 20, size=(4,)).astype(np.float32)
        self.current_step = 0

        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Random new cars arrive
        new_cars = np.random.randint(0, 5, size=(4,))
        self.state += new_cars

        # Cars pass depending on green signal
        if action == 0:
            # North-South green
            self.state[0] = max(0, self.state[0] - 8)
            self.state[1] = max(0, self.state[1] - 8)
        else:
            # East-West green
            self.state[2] = max(0, self.state[2] - 8)
            self.state[3] = max(0, self.state[3] - 8)

        # Avoid unlimited growth
        self.state = np.clip(self.state, 0, 100)

        total_waiting = np.sum(self.state)

        # Reward: less waiting is better
        reward = -total_waiting

        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.state.astype(np.float32), reward, terminated, truncated, {}