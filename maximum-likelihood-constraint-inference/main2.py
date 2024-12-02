import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv


# Neural network model
class TransitionLikelihoodNN(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        print(input_dim)
        super(TransitionLikelihoodNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


# Feature mapping function
def feature_mapping(st, st_next):
    return np.hstack([st, st_next])


# Reward function
def reward_function(st, st_next, allowed_end_state, banned_start_state):
    if st_next in allowed_end_state and st not in banned_start_state:
        return 1
    return 0


# Load and process trajectory data
def load_and_process_trajectories(filepath, n, allowed_end_state, banned_start_state):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Trajectory data not found! Ensure the file exists.")

    with open(filepath, 'rb') as handle:
        trajs = pickle.load(handle)

    processed_trajs = []
    for traj in trajs:
        new_traj = [item[0] + item[1] * n for item in traj]
        if len(new_traj) > 1 and new_traj[0] not in banned_start_state and new_traj[-1] in allowed_end_state:
            processed_trajs.append(new_traj)
    print(f"Found {len(processed_trajs)} valid trajectories!")
    return processed_trajs


# Training loop
def train_model(model, optimizer, transitions, epochs, allowed_end_state, banned_start_state):
    criterion = nn.BCELoss()
    reward_history = []
    success_rate_history = []
    violation_rate_history = []

    for epoch in range(epochs):
        total_loss = 0
        total_reward = 0
        success_count = 0
        violation_count = 0

        for traj in transitions:
            for i in range(len(traj) - 1):
                st = traj[i]
                st_next = traj[i + 1]
                features = torch.tensor(feature_mapping(st, st_next), dtype=torch.float32)
                label = torch.tensor(
                    [1.0 if reward_function(st, st_next, allowed_end_state, banned_start_state) == 1 else 0.0])

                prob = model(features)
                loss = criterion(prob, label)
                total_loss += loss.item()

                # Metrics
                reward = reward_function(st, st_next, allowed_end_state, banned_start_state)
                total_reward += reward
                success_count += int(reward == 1)
                violation_count += int(reward == 0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_reward = total_reward / len(transitions)
        success_rate = success_count / len(transitions)
        violation_rate = violation_count / len(transitions)

        reward_history.append(avg_reward)
        success_rate_history.append(success_rate)
        violation_rate_history.append(violation_rate)

        print(f"Epoch {epoch + 1}/{epochs}: Loss = {total_loss:.4f}, Avg Reward = {avg_reward:.4f}, "
              f"Success Rate = {success_rate:.4f}, Violation Rate = {violation_rate:.4f}")

    return reward_history, success_rate_history, violation_rate_history


# Save metrics to CSV
def save_metrics_to_csv(filename, header, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for epoch, value in enumerate(data, start=1):
            writer.writerow([epoch, value])


# Plot metrics
def plot_metric(metric_history, ylabel, title, filename):
    plt.plot(range(1, len(metric_history) + 1), metric_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# Main script
if __name__ == "__main__":
    n = 35
    allowed_end_state = [945, 946, 947, 948, 980, 981, 982, 983, 1015, 1016, 1017, 1018, 1050, 1051, 1052, 1053]
    banned_start_state = [1087]
    filepath = "pickles/trajectories.pickle"

    trajs = load_and_process_trajectories(filepath, n, allowed_end_state, banned_start_state)
    input_dim = 2 * n  # Combined dimensionality of st and st_next
    epochs = 50

    model = TransitionLikelihoodNN(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    reward_history, success_rate_history, violation_rate_history = train_model(
        model, optimizer, trajs, epochs, allowed_end_state, banned_start_state
    )

    # Save metrics to CSV
    save_metrics_to_csv('reward_history.csv', ["Epoch", "Average Reward"], reward_history)
    save_metrics_to_csv('success_rate_history.csv', ["Epoch", "Success Rate"], success_rate_history)
    save_metrics_to_csv('violation_rate_history.csv', ["Epoch", "Violation Rate"], violation_rate_history)

    # Plot metrics
    plot_metric(reward_history, "Average Reward", "Average Reward over Epochs", "reward.png")
    plot_metric(success_rate_history, "Success Rate", "Success Rate over Epochs", "success_rate.png")
    plot_metric(violation_rate_history, "Violation Rate", "Violation Rate over Epochs", "violation_rate.png")
