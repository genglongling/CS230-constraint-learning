# import numpy as np
# from mdp import BeamMDP
# from utils import calculate_kl_divergence, find_unaccrued_features
# import pickle
# import matplotlib.pyplot as plt
# import os, argparse, shutil
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--multi_goal", action="store_true")
# parser.add_argument("--do_constraint_inference", action="store_true")
# parser.add_argument("--policy_plot", action="store_true")
# parser.add_argument("--show_new_demos", action="store_true")
# args = parser.parse_args()
#
# n = 35 # 20 # dimensionality of state-space
# allowed_end_state = [945,946,947,948,980,981,982,983,1015,1016,1017,1018,1050,1051,1052,1053] # [320]
# banned_start_state = [1087] # [361]
#
# try:
#     if not os.path.exists("pickles"):
#         os.mkdir("pickles")
#     with open('pickles/trajectories.pickle', 'rb') as handle:
#         trajs = pickle.load(handle)
# except:
#     print("Cannot find trajectory data! Make sure pickles/trajectories.pickle exists.")
#     exit(0)
#
# new_trajs = []
# action_trajs = []
# stationary = []
# D0 = {}
# Dn = {}
# lens = []
# for traj in trajs:
#     new_traj = []
#     action_traj = []
#     for item in traj:
#         new_traj += [item[0]+item[1]*n]
#     for i in range(len(traj)-1):
#         dx = traj[i+1][0] - traj[i][0]
#         dy = traj[i+1][1] - traj[i][1]
#         if dx == -1 and dy == 0:
#             action_traj += [0]
#         if dx == 1 and dy == 0:
#             action_traj += [1]
#         if dx == 0 and dy == -1:
#             action_traj += [2]
#         if dx == 0 and dy == 1:
#             action_traj += [3]
#         if dx == -1 and dy == -1:
#             action_traj += [4]
#         if dx == 1 and dy == -1:
#             action_traj += [5]
#         if dx == -1 and dy == 1:
#             action_traj += [6]
#         if dx == 1 and dy == 1:
#             action_traj += [7]
#     action_traj += [8]
#     if len(new_traj) in [1]:
#         stationary += [new_traj]
#     elif (not args.multi_goal) and (new_traj[-1] not in allowed_end_state) or (new_traj[0] in banned_start_state):
#         pass
#     else:
#         lens += [len(new_traj)]
#         if new_traj[0] not in D0.keys():
#             D0[new_traj[0]] = 0
#         D0[new_traj[0]] += 1
#         if new_traj[-1] not in Dn.keys():
#             Dn[new_traj[-1]] = 0
#         Dn[new_traj[-1]] += 1
#         new_trajs += [np.array(new_traj).reshape(-1, 1)]
#         # print(new_trajs[-1].reshape(-1))
#         action_trajs += [np.array(action_traj).reshape(-1, 1)]
# trajs = new_trajs
# print("Found %d trajectories!" % len(trajs))
#
# if args.show_new_demos:
#     try:
#         if not os.path.exists("pickles"):
#             os.mkdir("pickles")
#         with open('pickles/new_demonstrations2.pickle', 'rb') as handle:
#             new_trajs = pickle.load(handle)
#     except:
#         print("Cannot find new demonstrations data! Make sure pickles/new_demonstrations.pickle exists.")
#         exit(0)
#
#     for traj in new_trajs:
#         print(traj.reshape(-1))
#     exit(0)
#
# if args.policy_plot:
#     if os.path.exists("figures"):
#         shutil.rmtree("figures")
#     os.mkdir("figures")
#
# def action_plot(grid, obstacles, iternum):
#     w = 0.8
#     if n > 20:
#         plt.figure(figsize=(12.8, 9.6))
#     gridmax = np.max(grid[:, :], axis=0)
#     for s in range(n*n):
#         i, j = int(s)%n, int(s)//n
#         up, down, left, right, ul, dl, ur, dr, null = grid[s, :]
#         # Rotate by 90 degrees
#         left, right, down, up, dl, dr, ul, ur = up, down, left, right, ul, dl, ur, dr
#         if s in obstacles:
#             color = 'brown'
#         elif s in D0.keys(): # initial state
#             color = 'pink'
#         elif null > 0:
#             color = 'blue'
#         else:
#             color = 'silver'
#         plt.plot([i,i,i+w,i+w,i], [j,j+w,j+w,j,j], color=color)
#         if s in obstacles or s in D0.keys():
#             continue
#         plt.plot([i+w/2,i+w/2], [j+w/2,j+w/2+w/2*up], color='red')
#         plt.plot([i+w/2,i+w/2], [j+w/2,j+w/2-w/2*down], color='red')
#         plt.plot([i+w/2,i+w/2-w/2*left], [j+w/2,j+w/2], color='red')
#         plt.plot([i+w/2,i+w/2+w/2*right], [j+w/2,j+w/2], color='red')
#         plt.plot([i+w/2,i+w/2-w/2*ul], [j+w/2,j+w/2+w/2*ul], color='green')
#         plt.plot([i+w/2,i+w/2-w/2*dl], [j+w/2,j+w/2-w/2*dl], color='green')
#         plt.plot([i+w/2,i+w/2+w/2*ur], [j+w/2,j+w/2+w/2*ur], color='green')
#         plt.plot([i+w/2,i+w/2+w/2*dr], [j+w/2,j+w/2-w/2*dr], color='green')
#     plt.xlabel("%d" % iternum)
#     plt.savefig("figures/%d.png" % iternum)
#
#
# nS = n*n; nA = 9; nF = 2;
# allow_diagonal_transitions = True
# feature_map = np.zeros((nS, nA, nF))
# feature_map[:, [0,1,2,3], 0] += 1 # distance
# feature_map[:, [4,5,6,7], 0] += np.sqrt(2) # distance
# feature_map[stationary, :, 1] += 1 # stationary
# feature_wts = np.array([-0.1, 0])
#
# # **Reward history list initialization** - New list to track rewards
# reward_history = []  # <-- NEW CODE
# success_rate_history = []
# violation_rate_history = []
# state_dim = n; discount = 1.0; infinite_horizon = False;
#
# #
# # if args.do_constraint_inference:
# #
# #     print("nominal mdp")
# #     state_dim = n; discount = 1.0; infinite_horizon = False;
# #     nominal_mdp = BeamMDP(nS, Dn, D0, state_dim, [], [], infinite_horizon, discount, feature_map, feature_wts, allow_diagonal_transitions)
# #
# #     print("generate demos")
# #     state_seqs, action_seqs = trajs, action_trajs
# #     demonstrations = [state_seqs, action_seqs]
# #
# #     # 200
# #     n_constraints = 200; kl_div_eps = 0.0001; demos = 100;
# #     nominal_mdp.backward_pass(max(lens))
# #     nominal_mdp.forward_pass(max(lens))
# #
# #     if args.policy_plot:
# #         action_plot(nominal_mdp.local_action_probs[:, :, 0], nominal_mdp.obstacles+nominal_mdp.state_constraints, 101-1)
# #     unaccrued_features = find_unaccrued_features(demonstrations, nominal_mdp)
# #     estimated_mdp = BeamMDP(nominal_mdp)
# #     likely_constraints = np.zeros((n_constraints))
# #     eliminated_trajectories = np.zeros((n_constraints))
# #     d_kl_history = np.zeros((n_constraints+1))
# #     d_kl_history[0] = calculate_kl_divergence(state_seqs, estimated_mdp)
# #
# #     for i in range(n_constraints):
# #         print("Iteration", i)
# #         fah = estimated_mdp.feature_accrual_history[unaccrued_features, -1]
# #         max_index = np.argmax(fah)
# #         eliminated_trajectories[i] = fah[max_index]
# #
# #         if eliminated_trajectories[i] == 0:
# #             d_kl_history[i+1] = d_kl_history[i]
# #             continue
# #
# #         likely_constraints[i] = unaccrued_features[max_index]
# #         estimated_mdp = BeamMDP(estimated_mdp, likely_constraints[i:i+1], True)
# #         estimated_mdp.backward_pass(max(lens))
# #         estimated_mdp.forward_pass(max(lens))
# #         current_reward = estimated_mdp.get_reward()
# #         reward_history.append(current_reward)
# #         current_sr = estimated_mdp.get_success_rate()
# #         success_rate_history.append(current_sr)
# #         print(success_rate_history)
# #         current_vr = estimated_mdp.get_violation_rate()
# #         violation_rate_history.append(current_vr)
# #         print(violation_rate_history)
# #
# #         if not os.path.exists("pickles"):
# #             os.mkdir("pickles")
# #         with open("pickles/new_demonstrations.pickle", 'wb') as handle:
# #             pickle.dump(estimated_mdp.produce_demonstrations(max(lens), demos)[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
# #         if args.policy_plot:
# #             action_plot(estimated_mdp.local_action_probs[:, :, 0], estimated_mdp.obstacles+estimated_mdp.state_constraints, 101+i)
# #
# #         d_kl_history[i+1] = calculate_kl_divergence(state_seqs, estimated_mdp)
# #
# #         if np.abs(d_kl_history[i+1]-d_kl_history[i]) < kl_div_eps:
# #             print("stopping")
# #             estimated_mdp = BeamMDP(nominal_mdp, likely_constraints[:i+1], True)
# #             estimated_mdp.backward_pass(max(lens))
# #             estimated_mdp.forward_pass(max(lens))
# #             if not os.path.exists("pickles"):
# #                 os.mkdir("pickles")
# #             with open("pickles/new_demonstrations.pickle", 'wb') as handle:
# #                 pickle.dump(estimated_mdp.produce_demonstrations(max(lens), demos)[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
# #             break
# #
# #     constraints = {
# #         'state': estimated_mdp.state_constraints,
# #         'action': estimated_mdp.action_constraints,
# #         'feature': estimated_mdp.feature_constraints,
# #     }
# #     if not os.path.exists("pickles"):
# #         os.mkdir("pickles")
# #     with open('pickles/constraints2.pickle', 'wb') as handle:
# #         pickle.dump(constraints, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# # Initialize beam search parameters
# beam_width = 5
# max_iterations = 200
# kl_div_eps = 0.0001
#
#
# # Function for KL Divergence calculation
# # Function for KL Divergence calculation
# # Function for KL Divergence calculation
# def calculate_kl_divergence(true_seqs, estimated_mdp):
#     kl_div = 0
#     for traj in true_seqs:
#         for state in traj:
#             state_idx = state[0]  # Assuming the state is a 1D array, adapt as needed
#
#             # Check if local_action_probs is valid for the state
#             if estimated_mdp.local_action_probs[state_idx] is not None:
#                 action_probs = estimated_mdp.local_action_probs[state_idx]
#
#                 # Check if the action probabilities are not empty
#                 if action_probs and isinstance(action_probs, list):
#                     action_prob = np.max(action_probs)  # Using the max probability for simplicity
#
#                     # Ensure we do not log zero probabilities
#                     if action_prob > 0:
#                         kl_div += np.log(action_prob)
#                 else:
#                     # If action_probs is empty, you can either skip this state or handle it in some way
#                     print(f"Warning: action_probs is empty or invalid for state {state_idx}")
#             else:
#                 # Handle the case where local_action_probs is None for the state
#                 print(f"Warning: local_action_probs is None for state {state_idx}")
#
#     return kl_div
#
#
# # Function to find unaccrued features
# def find_unaccrued_features(demonstrations, estimated_mdp):
#     # Placeholder function; implement your logic for finding unaccrued features
#     # This function should compare the feature usage in the demonstrations with the estimated MDP
#     return np.random.choice(range(10), size=5, replace=False)  # Example: Random selection
#
# # Beam search procedure
# beam = [(traj, action_traj, 1.0) for traj, action_traj in zip(new_trajs, action_trajs)]  # Initialize with likelihood = 1.0
#
# for iteration in range(max_iterations):
#     # Sort candidates based on likelihood (highest first)
#     beam = sorted(beam, key=lambda x: x[2], reverse=True)[:beam_width]
#
#     # Update the reward history and other metrics
#     current_reward = np.mean([x[2] for x in beam])  # average reward in the beam
#     reward_history.append(current_reward)
#
#     # Success rate and violation rate (placeholders)
#     current_success_rate = np.random.random()  # placeholder logic
#     success_rate_history.append(current_success_rate)
#
#     # KL divergence calculation
#     estimated_mdp = BeamMDP(nS, Dn, D0, state_dim, [], [], infinite_horizon, discount, feature_map, feature_wts, allow_diagonal_transitions)
#     d_kl = calculate_kl_divergence(new_trajs, estimated_mdp)
#     print(f"Iteration {iteration}, KL Divergence: {d_kl}")
#
#     # If KL divergence converges, break the loop
#     if iteration > 0 and np.abs(reward_history[-1] - reward_history[-2]) < kl_div_eps:
#         print(f"Stopping after {iteration} iterations")
#         break
#
#     # Update the MDP based on new constraints from unaccrued features
#     unaccrued_features = find_unaccrued_features([new_trajs, action_trajs], estimated_mdp)
#     for feature in unaccrued_features:
#         estimated_mdp.add_constraint(feature)
#
#     # Save intermediate demonstrations
#     if not os.path.exists("pickles"):
#         os.mkdir("pickles")
#     with open("pickles/new_demonstrations2.pickle", 'wb') as handle:
#         pickle.dump([x[0] for x in beam], handle, protocol=pickle.HIGHEST_PROTOCOL)

import numpy as np
from mdp import BeamMDP
from utils import find_unaccrued_features  # Removed KL divergence import
import pickle
import matplotlib.pyplot as plt
import os, argparse, shutil

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("--multi_goal", action="store_true")
parser.add_argument("--do_constraint_inference", action="store_true")
parser.add_argument("--policy_plot", action="store_true")
parser.add_argument("--show_new_demos", action="store_true")
args = parser.parse_args()

n = 35  # Dimensionality of state-space
allowed_end_state = [945,946,947,948,980,981,982,983,1015,1016,1017,1018,1050,1051,1052,1053]
banned_start_state = [1087]

# Load trajectory data
try:
    if not os.path.exists("pickles"):
        os.mkdir("pickles")
    with open('pickles/trajectories.pickle', 'rb') as handle:
        trajs = pickle.load(handle)
except:
    print("Cannot find trajectory data! Make sure pickles/trajectories.pickle exists.")
    exit(0)

new_trajs = []
action_trajs = []
stationary = []
D0 = {}
Dn = {}
lens = []

for traj in trajs:
    new_traj = []
    action_traj = []
    for item in traj:
        new_traj += [item[0]+item[1]*n]
    for i in range(len(traj)-1):
        dx = traj[i+1][0] - traj[i][0]
        dy = traj[i+1][1] - traj[i][1]
        if dx == -1 and dy == 0:
            action_traj += [0]
        if dx == 1 and dy == 0:
            action_traj += [1]
        if dx == 0 and dy == -1:
            action_traj += [2]
        if dx == 0 and dy == 1:
            action_traj += [3]
        if dx == -1 and dy == -1:
            action_traj += [4]
        if dx == 1 and dy == -1:
            action_traj += [5]
        if dx == -1 and dy == 1:
            action_traj += [6]
        if dx == 1 and dy == 1:
            action_traj += [7]
    action_traj += [8]
    if len(new_traj) in [1]:
        stationary += [new_traj]
    elif (not args.multi_goal) and (new_traj[-1] not in allowed_end_state) or (new_traj[0] in banned_start_state):
        pass
    else:
        lens += [len(new_traj)]
        if new_traj[0] not in D0.keys():
            D0[new_traj[0]] = 0
        D0[new_traj[0]] += 1
        if new_traj[-1] not in Dn.keys():
            Dn[new_traj[-1]] = 0
        Dn[new_traj[-1]] += 1
        new_trajs += [np.array(new_traj).reshape(-1, 1)]
        action_trajs += [np.array(action_traj).reshape(-1, 1)]
trajs = new_trajs
print("Found %d trajectories!" % len(trajs))

# Plot new demonstrations
if args.show_new_demos:
    try:
        if not os.path.exists("pickles"):
            os.mkdir("pickles")
        with open('pickles/new_demonstrations2.pickle', 'rb') as handle:
            new_trajs = pickle.load(handle)
    except:
        print("Cannot find new demonstrations data! Make sure pickles/new_demonstrations.pickle exists.")
        exit(0)

    for traj in new_trajs:
        print(traj.reshape(-1))
    exit(0)

if args.policy_plot:
    if os.path.exists("figures"):
        shutil.rmtree("figures")
    os.mkdir("figures")

# Reward and metrics history initialization
reward_history = []
success_rate_history = []
violation_rate_history = []

# Beam search parameters
beam_width = 5
max_iterations = 200

# Initialize the BeamMDP
nS = n*n; nA = 9; nF = 2
allow_diagonal_transitions = True
feature_map = np.zeros((nS, nA, nF))
feature_map[:, [0,1,2,3], 0] += 1  # distance
feature_map[:, [4,5,6,7], 0] += np.sqrt(2)  # distance
feature_map[stationary, :, 1] += 1  # stationary
feature_wts = np.array([-0.1, 0])
state_dim = n; discount = 1.0; infinite_horizon = False;

# Function to plot action probabilities
def action_plot(grid, obstacles, iternum):
    w = 0.8
    if n > 20:
        plt.figure(figsize=(12.8, 9.6))
    gridmax = np.max(grid[:, :], axis=0)
    for s in range(n*n):
        i, j = int(s)%n, int(s)//n
        up, down, left, right, ul, dl, ur, dr, null = grid[s, :]
        left, right, down, up, dl, dr, ul, ur = up, down, left, right, ul, dl, ur, dr
        if s in obstacles:
            color = 'brown'
        elif s in D0.keys():
            color = 'pink'
        elif null > 0:
            color = 'blue'
        else:
            color = 'silver'
        plt.plot([i,i,i+w,i+w,i], [j,j+w,j+w,j,j], color=color)
        if s in obstacles or s in D0.keys():
            continue
        plt.plot([i+w/2,i+w/2], [j+w/2,j+w/2+w/2*up], color='red')
        plt.plot([i+w/2,i+w/2], [j+w/2,j+w/2-w/2*down], color='red')
        plt.plot([i+w/2,i+w/2-w/2*left], [j+w/2,j+w/2], color='red')
        plt.plot([i+w/2,i+w/2+w/2*right], [j+w/2,j+w/2], color='red')
        plt.plot([i+w/2,i+w/2-w/2*ul], [j+w/2,j+w/2+w/2*ul], color='green')
        plt.plot([i+w/2,i+w/2-w/2*dl], [j+w/2,j+w/2-w/2*dl], color='green')
        plt.plot([i+w/2,i+w/2+w/2*ur], [j+w/2,j+w/2+w/2*ur], color='green')
        plt.plot([i+w/2,i+w/2+w/2*dr], [j+w/2,j+w/2-w/2*dr], color='green')
    plt.xlabel("%d" % iternum)
    plt.savefig("figures/%d.png" % iternum)


# Beam search procedure
beam = [(traj, action_traj, 1.0) for traj, action_traj in zip(new_trajs, action_trajs)]  # Initialize with likelihood = 1.0

for iteration in range(max_iterations):
    # Sort candidates based on likelihood (highest first)
    beam = sorted(beam, key=lambda x: x[2], reverse=True)[:beam_width]

    for state, action, _ in beam:
        estimated_mdp = BeamMDP(nS, Dn, D0, state_dim, [], [], False, 1.0, feature_map, feature_wts, allow_diagonal_transitions)
        estimated_mdp.backward_pass(max(lens))
        estimated_mdp.forward_pass(max(lens))
        # Update the reward history and other metrics
        current_reward = estimated_mdp.get_reward() #np.mean([x[2] for x in beam])  # average reward in the beam
        reward_history.append(current_reward)

        # Success rate and violation rate (placeholders)
        current_success_rate = estimated_mdp.get_success_rate()  # placeholder logic
        success_rate_history.append(current_success_rate)

        current_violation_rate = estimated_mdp.get_success_rate()  # placeholder logic
        violation_rate_history.append(current_violation_rate)


    # Save intermediate demonstrations
    if not os.path.exists("pickles"):
        os.mkdir("pickles")
    with open("pickles/new_demonstrations2.pickle", 'wb') as handle:
        pickle.dump([x[0] for x in beam], handle, protocol=pickle.HIGHEST_PROTOCOL)

# metric 1: average reward
# **Plot the reward history** - Plotting average reward across epochs
import csv

# Save the reward_history list to a CSV file
with open('reward_history2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Average Reward"])  # Optionally add a header row
    for epoch, reward in enumerate(reward_history, start=1):
        writer.writerow([epoch, reward])

# Save the reward_history list to a CSV file
with open('success_rate_history2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Success Rate History"])  # Optionally add a header row
    for epoch, sr in enumerate(success_rate_history, start=1):
        writer.writerow([epoch, sr])

# Save the reward_history list to a CSV file
with open('violation_rate_history2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Violation Rate History"])  # Optionally add a header row
    for epoch, vr in enumerate(violation_rate_history, start=1):
        writer.writerow([epoch, vr])

plt.plot(range(1, len(reward_history) + 1), reward_history, label="Average Reward", color='blue', marker='o')  # <-- NEW CODE
plt.xlabel('Epoch')  # <-- NEW CODE
plt.ylabel('Average Reward')  # <-- NEW CODE
plt.title('Average Reward over Epochs')  # <-- NEW CODE
plt.grid(True)  # <-- NEW CODE
plt.legend()  # <-- NEW CODE
plt.savefig('reward2.png')  # <-- SAVE PLOT
plt.show()  # <-- NEW CODE
