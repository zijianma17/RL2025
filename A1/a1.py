import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("plot_python_version")

from plot_python_version.src.grid_world import GridWorld

# =============================================================================
# 1. Setting up envrionment
# =============================================================================

ENV_SIZE = (5, 3)
NUM_STATES = ENV_SIZE[0] * ENV_SIZE[1] # 3*5

# Action space defined according to the provided grid_world ploting code
ACTION_SPACE = [
    (0, 1), # y+1 Down
    (1, 0), # x+1 Right
    (0, -1),# y-1 Up
    (-1, 0),# x-1 Left
    (0, 0)  # Stay
]
NUM_ACTIONS = len(ACTION_SPACE)

ACTION_STR_DICT = {
    (0, 1): "Down",
    (1, 0): "Right",
    (0, -1): "Up",
    (-1, 0): "Left",
    (0, 0): "Stay"    
}

# Special states (Only for ASSIGNMENT 1!!!)
# 先列后行, 从0开始
TARGET_STATE = (2, 2)
# FORBIDDEN_STATES = [(2, 1), (1, 2), (1, 3)]
FORBIDDEN_STATES = [(1,2), (2,1), (3,1)]

# discount_factor
GAMMA = 0.9

# reward setting
REWARD_TARGET = 1.0
REWARD_FORBIDDEN = -1.0
REWARD_BOUNDARY = -1.0
REWARD_STEP = 0.0 

# =============================================================================
# 2. Policy defination
# =============================================================================

# ======================
# State Utilities
# ======================

# s1, s2, s3, s4, s5
# s6, s7, fo, fo, s8
# s9, fo, s10(target), s11, s12
STATE_INDEX_LIST = [
    (0,0), (1,0), (2,0), (3,0), (4,0),
    (0,1), (1,1), (4,1),
    (0,2), (2,2), (3,2), (4,2)
]
def index_to_state(state_idx):
    return STATE_INDEX_LIST[state_idx-1]

# def state_to_index(state_coord):
#     return STATE_INDEX_LIST.index(state_coord) + 1

def idx_to_whole_idx(state_idx):
    idx_transform_dict = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
        6: 5, 7: 6, 8: 9,
        9: 10, 10: 12, 11: 13, 12: 14
    }
    return idx_transform_dict[state_idx]

def state_to_whole_idx(state_coord):
    idx = STATE_INDEX_LIST.index(state_coord) + 1
    return idx_to_whole_idx(idx)

def create_policy(policy_type: str = 'deterministic', forbidden_states=FORBIDDEN_STATES):
    """
    Define policy manually, 
    """
    # policy_matrix = np.zeros((NUM_STATES, NUM_ACTIONS))
    policy_matrix = np.random.rand(NUM_STATES, NUM_ACTIONS)
    # normalize to make each row sum to 1
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]
    # action_space 1~5 correspond to [Down, Right, Up, Left, Stay]

    def set_policy(state_idx, action_prob_list):
        policy_matrix[state_idx] = action_prob_list
        return

    # --- Deterministic Policy ---
    if policy_type == 'deterministic':
        """
        s1 s2 s3 s4 s5: R, R, R, R, D
        s6 s7 (forbid, forbid) s8: U, U, D
        s9 (forbid) s10, s11, s12: U, Stay, L, L
        """
        set_policy(idx_to_whole_idx(1), [0, 1, 0, 0, 0]) # s1: R
        set_policy(idx_to_whole_idx(2), [0, 1, 0, 0, 0]) # s2: R
        set_policy(idx_to_whole_idx(3), [0, 1, 0, 0, 0]) # s3: R
        set_policy(idx_to_whole_idx(4), [0, 1, 0, 0, 0]) # s4: R
        set_policy(idx_to_whole_idx(5), [1, 0, 0, 0, 0]) # s5: D
        set_policy(idx_to_whole_idx(6), [0, 0, 1, 0, 0]) # s6: U
        set_policy(idx_to_whole_idx(7), [0, 0, 1, 0, 0]) # s7: U
        set_policy(idx_to_whole_idx(8), [1, 0, 0, 0, 0]) # s8: D
        set_policy(idx_to_whole_idx(9), [0, 0, 1, 0, 0]) # s9: U
        set_policy(idx_to_whole_idx(10),[0, 0, 0, 0, 1]) # s10: Stay
        set_policy(idx_to_whole_idx(11),[0, 0, 0, 1, 0]) # s11: L
        set_policy(idx_to_whole_idx(12),[0, 0, 0, 1, 0]) # s12: L

    # --- Stochastic Policy ---
    elif policy_type == 'stochastic':
        """
        s1 s2 s3 s4 s5: R, R, R, R, D
        s6 s7 (forbid, forbid) s8: U, U, D
        s9 (forbid) s10, s11, s12: U, Stay, L, L
        """
        def stochastic_probs(best_action_idx) -> list:
            """
            return a list of action probabilities
            the best action has 0.6 prob, others share 0.4
            """
            probs = [0.4 / (NUM_ACTIONS - 1)] * NUM_ACTIONS
            probs[best_action_idx] = 0.6
            return probs

        set_policy(idx_to_whole_idx(1), stochastic_probs(1)) # s1: R
        set_policy(idx_to_whole_idx(2), stochastic_probs(1)) # s2: R
        set_policy(idx_to_whole_idx(3), stochastic_probs(1)) # s3: R
        set_policy(idx_to_whole_idx(4), stochastic_probs(1)) # s4: R
        set_policy(idx_to_whole_idx(5), stochastic_probs(0)) # s5: D
        set_policy(idx_to_whole_idx(6), stochastic_probs(2)) # s6: U
        set_policy(idx_to_whole_idx(7), stochastic_probs(2)) # s7: U
        set_policy(idx_to_whole_idx(8), stochastic_probs(0)) # s8: D
        set_policy(idx_to_whole_idx(9), stochastic_probs(2)) # s9: U
        set_policy(idx_to_whole_idx(10),stochastic_probs(4)) # s10: Stay
        set_policy(idx_to_whole_idx(11),stochastic_probs(3)) # s11: L
        set_policy(idx_to_whole_idx(12),stochastic_probs(3)) # s12: L
    
    else:
        raise ValueError(f"Unknown Policy Type: {policy_type}")
    
    return policy_matrix


# =============================================================================
# 3. Simulation
# =============================================================================

def run_simulation(env, policy_matrix, start_state, num_steps=50):
    """
    Using the plot code's GridWorld environment to run a simulation
    """
    env.start_state = start_state
    env.reset()

    STEP_RENDER = False
    if STEP_RENDER:
        env.render()
    
    trajectory = []
    
    for _ in range(num_steps):
        current_state_coord = env.agent_state
        state_idx = state_to_whole_idx(current_state_coord)
        
        action_probs = policy_matrix[state_idx]
        action_idx = np.random.choice(NUM_ACTIONS, p=action_probs)
        action_coord = ACTION_SPACE[action_idx]
        
        next_state, reward, done, _ = env.step(action_coord)
        if STEP_RENDER:
            env.render()
            action_str = ACTION_STR_DICT[action_coord]
            print(f"State: {current_state_coord}, Action: {action_str}-{action_coord}, Reward: {reward}, Next State: {next_state}, Done: {done}")
            time.sleep(0.1)
        
        trajectory.append((current_state_coord, action_idx, reward))
        
        # if done:
        #     break
            
    return trajectory

def calculate_discounted_return(trajectory, gamma):
    """calculate the total discounted return of a trajectory"""
    total_return = 0.0
    for i, (_, _, reward) in enumerate(trajectory):
        total_return += (gamma ** i) * reward
    return total_return


# =============================================================================
# 4. Main function
# =============================================================================

if __name__ == "__main__":
    
    # simulated args
    class MockArgs:
        env_size = ENV_SIZE
        start_state = (0, 0)
        target_state = TARGET_STATE
        forbidden_states = FORBIDDEN_STATES
        action_space = ACTION_SPACE
        reward_target = REWARD_TARGET
        reward_forbidden = REWARD_FORBIDDEN
        reward_step = REWARD_STEP
        animation_interval = 0.1
        debug = False
        
    args = MockArgs()
    
    # init environment
    env = GridWorld(env_size=args.env_size,
                    start_state=args.start_state,
                    target_state=args.target_state,
                    forbidden_states=args.forbidden_states)
    
    # starting state setting
    start_idx = 1
    start_coord = index_to_state(start_idx)
    
    # 1. Deterministic Policy
    print("--- Deterministic Policy ---")
    policy_det = create_policy("deterministic", FORBIDDEN_STATES)
    trajectory_det = run_simulation(env, policy_det, start_coord, num_steps=50)
    return_det = calculate_discounted_return(trajectory_det, GAMMA)
    print(f"Starting from {start_coord}, Total Discounted Return of the Trajectory: {return_det:.4f}")
    
    # 2. Stochastic Policy
    print("\n--- Stochastic Policy ---")
    policy_sto = create_policy("stochastic", FORBIDDEN_STATES)
    trajectory_sto = run_simulation(env, policy_sto, start_coord, num_steps=50)
    return_sto = calculate_discounted_return(trajectory_sto, GAMMA)
    print(f"Starting from {start_coord}, Total Discounted Return of the Trajectory: {return_sto:.4f}")

    def visualization(plot_policy_type="deterministic"):
        # --- Visualization ---
        if plot_policy_type == "deterministic":
            policy_to_plot = policy_det
            trajectory_to_plot = trajectory_det
        elif plot_policy_type == "stochastic":
            policy_to_plot = policy_sto
            trajectory_to_plot = trajectory_sto

        # 1. reset env and add policy
        env.start_state = start_coord
        env.reset()
        env.render()
        env.add_policy(policy_to_plot)

        # output the policy
        # np.savetxt(f"A1/policy_{plot_policy_type}.csv", np.round(policy_to_plot, 2), delimiter=",")
        np.savetxt(f"A1/policy_{plot_policy_type}.csv", policy_to_plot, delimiter=",")
        # np.savetxt(f"A1/policy_{plot_policy_type}.txt", np.round(policy_to_plot,1), delimiter=" ")

        # output the policy plot
        env.canvas.savefig(f"A1/policy_{plot_policy_type}.png")

        
        # 2. running and render the trajectory
        for (state_coord, action_idx, reward) in trajectory_to_plot:
            env.step(ACTION_SPACE[action_idx])
        
        print("\n========== Final Render ==========")
        print("========== Close the plot window to exit ==========")
        env.render()

        # output the trajectory plot
        env.canvas.savefig(f"A1/trajectory_{plot_policy_type}.png")

        plt.show(block=True)
        # 清空 plt
        plt.clf()
    
    # visualization("deterministic")
    visualization("stochastic")
    

