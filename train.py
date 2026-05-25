from stable_baselines3 import DQN
from traffic_env import TrafficEnv
import matplotlib.pyplot as plt


env = TrafficEnv()

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=50000)

model.save("traffic_dqn_model")

print("Training complete. Model saved.")