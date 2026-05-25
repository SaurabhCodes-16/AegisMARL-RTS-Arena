from stable_baselines3 import DQN
from traffic_env import TrafficEnv


env = TrafficEnv()
model = DQN.load("traffic_dqn_model")

obs, info = env.reset()

total_reward = 0

for step in range(50):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(f"Step {step+1}")
    print(f"Action: {'North-South Green' if action == 0 else 'East-West Green'}")
    print(f"Queue: {obs}")
    print(f"Reward: {reward}")
    print("-" * 40)

    if terminated or truncated:
        break

print("Total Reward:", total_reward)