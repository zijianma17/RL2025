"""
Code todo:
1. import from A2
2. Value iteration
3. Optimal policy extraction
4. Visualization and export of the value iteration history
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("plot_python_version")

from plot_python_version.src.grid_world import GridWorld

# =============================================================================
# import setting from a2.py
# =============================================================================
from A2.a2 import (
    ENV_SIZE,
    ACTION_SPACE,
    FORBIDDEN_STATES,
    GAMMA,
    coord_transform,
    i_to_coord,
    coord_to_i,
    init_env,
)
# =============================================================================

def value_iteration(gamma=GAMMA, theta=1e-6, max_iterations=1000):
    # 1. Value initialization
    V = np.zeros(ENV_SIZE[0] * ENV_SIZE[1]) # initial V_0
    history_V = [] # storing the evolution of V

    # 2. Main loop
    for iteration in range(max_iterations):
        delta = 0
        V_new = np.copy(V)

        # 3. For all states
        for s_idx in range(len(V)):
            # Skip forbidden states
            if i_to_coord(s_idx) in FORBIDDEN_STATES:
                continue

            v_old = V[s_idx]
            q_values = np.zeros(len(ACTION_SPACE))

            # 4. For all available actions in this state
            for action_i, action in enumerate(ACTION_SPACE):
                s_coord = i_to_coord(s_idx)
                next_coord, reward = coord_transform(s_coord, action)
                next_s_idx = coord_to_i(next_coord)
                
                q_s_a = reward + gamma * V[next_s_idx] # Note the V should be V, not V_new
                q_values[action_i] = q_s_a


            # 5. Update the value function: V_{k+1}(s) = max_a Q(s,a)
            V_new[s_idx] = np.max(q_values)
            delta = max(delta, abs(v_old - V_new[s_idx]))

        V = V_new
        
        history_V.append(np.copy(V)) # for plotting

        if delta < theta: # when converged
            print(f"Value Iteration converged after {iteration+1} iterations.")
            break

    return V, history_V # Return the optimal value function and the history


def extract_optimal_policy(V, gamma=GAMMA):
    optimal_policy = np.zeros((ENV_SIZE[0] * ENV_SIZE[1], len(ACTION_SPACE)))
    
    for s_idx in range(len(V)):
        if i_to_coord(s_idx) in FORBIDDEN_STATES:
            continue

        q_values = np.zeros(len(ACTION_SPACE))
        for action_i, action in enumerate(ACTION_SPACE):
            s_coord = i_to_coord(s_idx)
            next_coord, reward = coord_transform(s_coord, action)
            next_s_idx = coord_to_i(next_coord)
            q_s_a = reward + gamma * V[next_s_idx]
            q_values[action_i] = q_s_a
        
        best_action_idx = np.argmax(q_values)
        optimal_policy[s_idx][best_action_idx] = 1.0 # Deterministic
        
    return optimal_policy

def get_valid_P(policy):
    valid_indices = [coord_to_i((y, x)) for x in range(ENV_SIZE[1]) for y in range(ENV_SIZE[0]) if (y, x) not in FORBIDDEN_STATES]
    policy_valid = policy[valid_indices]
    return policy_valid

if __name__ == "__main__":
    pass

    # 1. init env
    env = init_env()

    # 2. value iteration
    V_optimal, history_V = value_iteration()

    # 3. extract optimal policy
    optimal_policy = extract_optimal_policy(V_optimal)
    valid_optimal_policy = get_valid_P(optimal_policy)
    np.savetxt("A3/optimal_policy.csv", valid_optimal_policy, delimiter=",")
    # visualize the optimal policy
    env.add_policy(optimal_policy)

    # 4. visualize the final value
    env.add_state_values(V_optimal)
    env.ax.set_title("Optimal State Value Function from Value Iteration")
    env.render()
    env.canvas.savefig("A3/optimal_value_function.png")

    input("========== The optimal value function has been saved as 'A3/optimal_value_function.png'. Press Enter to continue... ==========")

    # 5. output the value iteration history of first 5 and last 5 iterations
    # erase the policy for clear visualization
    env = init_env()

    for i, V in enumerate(history_V):
        if i < 5 or i >= len(history_V) - 5:
            env.add_state_values(V)
            env.ax.set_title(f"State Value Function at Iteration {i+1}")
            env.render()
            # clip the margin
            env.canvas.savefig(f"A3/value_function_iteration_{i+1}.png", bbox_inches='tight', pad_inches=0.1)
            print(f"========== The value function at iteration {i+1} has been saved as 'A3/value_function_iteration_{i+1}.png'. ==========")

    print("\n\n========= All Done! Thanks for reviewing! ==========\n\n")
