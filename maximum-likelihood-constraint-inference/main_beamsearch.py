
import numpy as np
from mdp import GridMDP
from utils import calculate_kl_divergence, find_unaccrued_features
import pickle
import matplotlib.pyplot as plt
import os, argparse, shutil
from queue import PriorityQueue


parser = argparse.ArgumentParser()
parser.add_argument("--multi_goal", action="store_true")
parser.add_argument("--do_constraint_inference", action="store_true")
parser.add_argument("--policy_plot", action="store_true")
parser.add_argument("--show_new_demos", action="store_true")
args = parser.parse_args()

def beam_search_constraint_inference_with_trajectories(
        nominal_mdp, n_constraints, beam_width, max_traj_length, demos
):
    """
    Perform beam search to identify constraints and generate trajectories.

    Arguments:
    - nominal_mdp: The nominal MDP object.
    - n_constraints: Maximum number of constraints to infer.
    - beam_width: Maximum width of the beam.
    - max_traj_length: Maximum trajectory length for trajectory generation.
    - demos: Number of demonstrations to generate.

    Returns:
    - best_mdp: The MDP with the best constraints inferred by beam search.
    - reward_history: List of rewards across beam search iterations.
    - success_rate_history: List of success rates across beam search iterations.
    - violation_rate_history: List of violation rates across beam search iterations.
    - trajectories: Generated trajectories for the best MDP.
    """
    # Initialize beam with the nominal MDP (no constraints)
    beam = PriorityQueue()
    initial_d_kl = calculate_kl_divergence([], nominal_mdp)
    beam.put((initial_d_kl, []))

    best_mdp = nominal_mdp
    best_d_kl = initial_d_kl
    reward_history, success_rate_history, violation_rate_history = [], [], []

    for i in range(n_constraints):
        print(f"Beam search iteration {i + 1}/{n_constraints}")
        new_beam = PriorityQueue()

        while not beam.empty():
            current_d_kl, current_constraints = beam.get()
            estimated_mdp = GridMDP(nominal_mdp, current_constraints, True)
            estimated_mdp.backward_pass(max_traj_length)
            estimated_mdp.forward_pass(max_traj_length)

            # Generate feature accrual history
            unaccrued_features = find_unaccrued_features([], nominal_mdp)
            fah = estimated_mdp.feature_accrual_history[unaccrued_features, -1]

            # Explore adding constraints from unaccrued features
            for feature_idx in range(len(fah)):
                if fah[feature_idx] > 0:
                    new_constraints = current_constraints + [unaccrued_features[feature_idx]]
                    new_mdp = GridMDP(nominal_mdp, new_constraints, True)
                    new_mdp.backward_pass(max_traj_length)
                    new_mdp.forward_pass(max_traj_length)

                    # Evaluate KL divergence with new constraints
                    new_d_kl = calculate_kl_divergence([], new_mdp)
                    new_beam.put((new_d_kl, new_constraints))

                    if new_d_kl < best_d_kl:
                        best_d_kl = new_d_kl
                        best_mdp = new_mdp

                    # Collect performance metrics
                    reward_history.append(new_mdp.get_reward())
                    success_rate_history.append(new_mdp.get_success_rate())
                    violation_rate_history.append(new_mdp.get_violation_rate())

        # Select top candidates for the next beam
        candidates = []
        while not new_beam.empty() and len(candidates) < beam_width:
            candidates.append(new_beam.get())
        for candidate in candidates:
            beam.put(candidate)

        # Early stopping condition
        if len(candidates) == 0 or np.abs(candidates[0][0] - best_d_kl) < 1e-4:
            print("Beam search converged early.")
            break

    # Generate trajectories for the best MDP
    trajectories = best_mdp.produce_demonstrations(max_traj_length, demos)[0]

    return best_mdp, reward_history, success_rate_history, violation_rate_history, trajectories

import numpy as np

n = 35  # 20 # dimensionality of state-space
allowed_end_state = [945, 946, 947, 948, 980, 981, 982, 983, 1015, 1016, 1017, 1018, 1050, 1051, 1052,
                     1053]  # [320]
banned_start_state = [1087]  # [361]

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
        new_traj += [item[0] + item[1] * n]
    for i in range(len(traj) - 1):
        dx = traj[i + 1][0] - traj[i][0]
        dy = traj[i + 1][1] - traj[i][1]
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
        # print(new_trajs[-1].reshape(-1))
        action_trajs += [np.array(action_traj).reshape(-1, 1)]
trajs = new_trajs
print("Found %d trajectories!" % len(trajs))

nS = n * n;
nA = 9;
nF = 2;
allow_diagonal_transitions = True
feature_map = np.zeros((nS, nA, nF))
feature_map[:, [0, 1, 2, 3], 0] += 1  # distance
feature_map[:, [4, 5, 6, 7], 0] += np.sqrt(2)  # distance
feature_map[stationary, :, 1] += 1  # stationary
feature_wts = np.array([-0.1, 0])

# Replace these placeholders with your actual objects
print("nominal mdp")
state_dim = n;
discount = 1.0;
infinite_horizon = False;
nominal_mdp = GridMDP(nS, Dn, D0, state_dim, [], [], infinite_horizon, discount, feature_map, feature_wts,
                      allow_diagonal_transitions)

n_constraints = 10
beam_width = 5
max_traj_length = 50
demos = 100

# Perform beam search and generate trajectories
best_mdp, reward_history, success_rate_history, violation_rate_history, trajectories = beam_search_constraint_inference_with_trajectories(
    nominal_mdp, n_constraints, beam_width, max_traj_length, demos
)

# Save results and metrics
save_metrics_to_csv('reward_history.csv', reward_history, 'Reward')
save_metrics_to_csv('success_rate_history.csv', success_rate_history, 'Success Rate')
save_metrics_to_csv('violation_rate_history.csv', violation_rate_history, 'Violation Rate')

# Plot metrics
plot_metrics(reward_history, 'Reward History', 'Reward', 'reward_history.png')
plot_metrics(success_rate_history, 'Success Rate History', 'Success Rate', 'success_rate_history.png')
plot_metrics(violation_rate_history, 'Violation Rate History', 'Violation Rate', 'violation_rate_history.png')

# Output generated trajectories
print(f"Generated trajectories: {trajectories}")

