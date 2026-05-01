import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from config import RL_DATASET_PATH, DQN_MODEL_PATH, FIGURE_DIR, REPORT_DIR
from models.dqn_agent import DQN
from models.replay_memory import ReplayMemory

class TriageEnv:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)
        self.index = 0

    def reset(self):
        self.index = 0
        return self._get_state(self.df.iloc[self.index])

    def _get_state(self, row):
        return np.array([
            row["p_home"],
            row["p_doctor"],
            row["p_emergency"]
        ], dtype=np.float32)

    def step(self, action: int):
        row = self.df.iloc[self.index]
        correct_action = int(row["correct_action"])

        if action == correct_action:
            reward = 10.0
        elif correct_action == 2 and action != 2:
            reward = -15.0
        elif correct_action == 0 and action == 2:
            reward = -5.0
        else:
            reward = -3.0

        self.index += 1
        done = self.index >= len(self.df)

        if done:
            next_state = np.zeros(3, dtype=np.float32)
        else:
            next_state = self._get_state(self.df.iloc[self.index])

        return next_state, reward, done

def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)

    state_batch = torch.tensor(np.array([t.state for t in transitions]), dtype=torch.float32)
    action_batch = torch.tensor([t.action for t in transitions], dtype=torch.int64).unsqueeze(1)
    reward_batch = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    next_state_batch = torch.tensor(np.array([t.next_state for t in transitions]), dtype=torch.float32)
    done_batch = torch.tensor([t.done for t in transitions], dtype=torch.float32)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    with torch.no_grad():
        max_next_q = target_net(next_state_batch).max(1)[0]
        target_q = reward_batch + gamma * max_next_q * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    df = pd.read_csv(RL_DATASET_PATH)
    env = TriageEnv(df)

    state_dim = 3
    action_dim = 3

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(capacity=5000)

    gamma = 0.99
    batch_size = 32
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    episodes = 30

    rewards = []
    avg_losses = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = int(torch.argmax(policy_net(state_tensor), dim=1).item())

            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)

            loss = optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        rewards.append(total_reward)
        avg_losses.append(avg_loss)

        print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}")

    torch.save(policy_net.state_dict(), DQN_MODEL_PATH)
    print("Saved DQN model:", DQN_MODEL_PATH)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, episodes + 1), rewards, marker="o")
    plt.title("DQN Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/dqn_rewards.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, episodes + 1), avg_losses, marker="o")
    plt.title("DQN Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/dqn_losses.png")
    plt.close()

    with open(f"{REPORT_DIR}/dqn_report.txt", "w", encoding="utf-8") as f:
        f.write("Episode Rewards:\n")
        f.write(str(rewards))
        f.write("\n\nAverage Losses:\n")
        f.write(str(avg_losses))

if __name__ == "__main__":
    main()