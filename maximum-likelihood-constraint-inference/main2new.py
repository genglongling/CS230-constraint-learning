import numpy as np
import csv
import matplotlib.pyplot as plt


def store_to_csv(file_name, data):
    """
    Store the given data to a CSV file.

    Parameters:
    - file_name: The name of the CSV file to store the data.
    - data: The data to be written into the file (list or 2D array).
    """
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Value'])  # Write header
        for i, value in enumerate(data):
            writer.writerow([i + 1, value])  # Write iteration and value


def beam_search(feature_map, feature_weights, nS, nA, allowed_end_state, banned_start_state, beam_width=5,
                max_iterations=200):
    """
    Perform beam search to compute the following metrics:
    1. Reward
    2. Success Rate
    3. Violation Rate

    Parameters:
    - feature_map: Feature map containing the features for each state-action pair
    - feature_weights: Weights for each feature to compute the reward
    - nS: Number of states
    - nA: Number of actions
    - allowed_end_state: List of goal (end) states
    - banned_start_state: List of invalid starting states
    - beam_width: Number of top candidates to retain at each iteration
    - max_iterations: Maximum number of iterations for the search

    Returns:
    - reward_history: A list of rewards for the top candidate solutions at each step
    - success_rate_history: Success rate over the iterations
    - violation_rate_history: Violation rate over the iterations
    """

    reward_history = []
    success_rate_history = []
    violation_rate_history = []

    # Initialize the beam with the initial state (starting state)
    beam = [(0, np.zeros(nS), 0)]  # (cost, state vector, time)
    #print(beam)
    def is_goal_state(state):
        return state in allowed_end_state

    def is_violated_state(state):
        return state in banned_start_state

    def calculate_reward(state):
        # Assuming feature_map is of shape (nA, nS, nF)
        # Calculate the reward by summing over features based on the feature weights
        reward = 0
        for action in range(nA):
            # Get the feature vector for this action and state
            feature_vector = feature_map[action, state, :]  # Focus on a single state
            # Compute the reward contribution for this action
            reward += np.dot(feature_vector, feature_weights)
        return reward

    for iteration in range(max_iterations):
        new_beam = []

        # Generate the next states from the current beam
        for cost, state, time in beam:
            # Loop through all possible actions
            for action in range(nA):
                # Ensure action properly updates the state by accessing feature_map with state and action
                # Assuming feature_map is (nA, nS, nF), update the state using the appropriate feature map
                next_state = state.copy()  # Make a copy of the state to apply the transition

                # Update state based on feature map and action
                # Assuming feature_map is (nA, nS, nF), update the state using the appropriate feature map
                state_feature_change = feature_map[action, :, :]  # Get the feature change for the action

                # You need to apply this change to the state correctly
                next_state += np.sum(state_feature_change, axis=1)  # Example of updating state based on action

                # Calculate the reward for the next state-action pair
                reward = calculate_reward(next_state)
                new_cost = cost + reward  # Total cost = previous cost + reward

                # Append to new beam (state, action, cost, and time step)
                new_beam.append((new_cost, next_state, time + 1))

        print(new_beam)
        # Sort the new beam by cost (minimization of cost corresponds to maximization of reward)
        new_beam.sort(key=lambda x: x[0])

        # Retain only the top `beam_width` candidates
        beam = new_beam[:beam_width]

        # Track reward, success, and violation for each step in the beam
        rewards = [item[0] for item in beam]
        reward_history.append(np.mean(rewards))  # Mean reward at this iteration

        # Calculate success rate and violation rate
        success_count = 0
        violation_count = 0

        for _, state, _ in beam:
            if is_goal_state(state):
                success_count += 1
            if is_violated_state(state):
                violation_count += 1

        success_rate = success_count / len(beam)
        violation_rate = violation_count / len(beam)

        success_rate_history.append(success_rate)
        violation_rate_history.append(violation_rate)

    # Store the lists to files (CSV format)
    store_to_csv('reward_history2.csv', reward_history)
    store_to_csv('success_rate_history2.csv', success_rate_history)
    store_to_csv('violation_rate_history2.csv', violation_rate_history)

    return reward_history, success_rate_history, violation_rate_history


# Example usage:
# Initialize feature map, feature weights, and parameters
nS = 35 * 35  # State space size (e.g., grid world size)
nA = 9  # Number of actions
feature_map = np.zeros((nA, nS, 2))  # Placeholder for feature map (size: [nA, nS, nF])
feature_weights = np.array([-0.1, 0])  # Placeholder feature weights
allowed_end_state = [945, 946, 947, 948, 980, 981]  # Example goal states
banned_start_state = [1087]  # Example blocked start states

# Run beam search
reward_history, success_rate_history, violation_rate_history = beam_search(
    feature_map, feature_weights, nS, nA, allowed_end_state, banned_start_state, beam_width=5, max_iterations=200
)

# Plot the results
plt.plot(reward_history, label="Reward")
plt.plot(success_rate_history, label="Success Rate")
plt.plot(violation_rate_history, label="Violation Rate")
plt.legend()
plt.show()
