
"""
code todo list:
1. define r_pi, and P_pi
    手动编辑，r_pi = [reward]
    P_pi其实是probability transition matrix.
2. using closed-form solution to compute V_pi
    矩阵运算
3. develop iterative policy evaluation to compute V_pi

4. plot the state value without policy arrows

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("plot_python_version")

from plot_python_version.src.grid_world import GridWorld

# =============================================================================
# import setting from a1.py
# =============================================================================

from A1.a1 import (
    ENV_SIZE,
    NUM_STATES,
    ACTION_SPACE,
    NUM_ACTIONS,
    ACTION_STR_DICT,
    ACTION_STR_DICT_REVERSE,
    TARGET_STATE,
    FORBIDDEN_STATES,
    GAMMA,
    REWARD_TARGET,
    REWARD_FORBIDDEN,
    REWARD_BOUNDARY,
    REWARD_STEP,
    index_to_coord,
    idx_to_whole_idx,
    state_to_whole_idx,
    create_policy,   
)

def coord_transform(coord, action):
    next_coord = (coord[0] + action[0], coord[1] + action[1])
    if next_coord[0] < 0 or next_coord[0] >= ENV_SIZE[0] or \
         next_coord[1] < 0 or next_coord[1] >= ENV_SIZE[1]:
        return coord, REWARD_BOUNDARY
    elif next_coord in FORBIDDEN_STATES:
        return coord, REWARD_FORBIDDEN
    elif next_coord == TARGET_STATE:
        return next_coord, REWARD_TARGET
    else:
        return next_coord, REWARD_STEP
    
def i_to_coord(index):
    x = index % ENV_SIZE[0]
    y = index // ENV_SIZE[0]
    return (x, y)

def coord_to_i(coord):
    return coord[0] + coord[1] * ENV_SIZE[0]

def defining_r_pi_and_P_pi(policy_type:str="deterministic"):

    # 根据输入的policy, 遍历所有state(行), 
    # 遍历action(列), 
    # 根据action的对应得到对应的next_state,
    # 根据next_state得到对应的reward, 反向加到r_pi[state]
    # 根据next_state得到对应的index, 反向加到P_pi[state][next_state_index]
    policy = create_policy(policy_type=policy_type)
    num_coords = ENV_SIZE[0] * ENV_SIZE[1]
    r_pi = np.zeros(num_coords)
    P_pi = np.zeros((num_coords, num_coords))

    for i in range(policy.shape[0]):
        coord = i_to_coord(i)
        for action_idx, action_prob in enumerate(policy[i]):
            action = ACTION_SPACE[action_idx]
            next_coord, reward = coord_transform(coord, action)
            next_i = coord_to_i(next_coord)
            r_pi[i] += action_prob * reward
            P_pi[i][next_i] += action_prob
    
    return r_pi, P_pi

def get_valid_r_pi_and_P_pi(r_pi, P_pi):
    # extract the valid indices (not forbidden states)
    valid_indices = [coord_to_i((y, x)) for x in range(ENV_SIZE[1]) for y in range(ENV_SIZE[0]) if (y, x) not in FORBIDDEN_STATES]
    r_pi_valid = r_pi[valid_indices]
    P_pi_valid = P_pi[np.ix_(valid_indices, valid_indices)]
    return r_pi_valid, P_pi_valid


def closed_form_solution(r_pi, P_pi, gamma=GAMMA):
    I = np.eye(P_pi.shape[0])
    V_pi = np.linalg.inv(I - gamma * P_pi).dot(r_pi)
    return V_pi


def iterative_policy_evaluation(r_pi, P_pi, gamma=GAMMA, theta=1e-6, max_iterations=1000):
    V_pi = np.zeros(r_pi.shape)
    for iteration in range(max_iterations):
        delta = 0
        for s in range(len(r_pi)):
            v = V_pi[s]
            V_pi[s] = r_pi[s] + gamma * np.sum(P_pi[s] * V_pi)
            delta = max(delta, abs(v - V_pi[s]))
        if delta < theta:
            print(f"Converged after {iteration+1} iterations.")
            break
    return V_pi

def init_env():
    class MockArgs:
        env_size = ENV_SIZE
        start_state = (0, 0)
        target_state = TARGET_STATE
        forbidden_states = FORBIDDEN_STATES
        action_space = ACTION_SPACE
        reward_target = REWARD_TARGET
        reward_forbidden = REWARD_FORBIDDEN
        reward_step = REWARD_STEP
        debug = False
    args = MockArgs()
    
    # init environment
    env = GridWorld(env_size=args.env_size,
                    start_state=args.start_state,
                    target_state=args.target_state,
                    forbidden_states=args.forbidden_states)
    
    env.reset()
    env.render()
    return env

if __name__ == "__main__":

    # 1. Deterministic Policy
    r_pi_det, P_pi_det = defining_r_pi_and_P_pi("deterministic")
    r_pi_det_valid, P_pi_det_valid = get_valid_r_pi_and_P_pi(r_pi_det, P_pi_det)
    np.savetxt("A2/r_pi_deterministic.csv", r_pi_det_valid, delimiter=",")
    np.savetxt("A2/P_pi_deterministic.csv", P_pi_det_valid, delimiter=",")


    # 2. Stochastic Policy
    r_pi_sto, P_pi_sto = defining_r_pi_and_P_pi("stochastic")
    r_pi_sto_valid, P_pi_sto_valid = get_valid_r_pi_and_P_pi(r_pi_sto, P_pi_sto)
    np.savetxt("A2/r_pi_stochastic.csv", r_pi_sto_valid, delimiter=",")
    np.savetxt("A2/P_pi_stochastic.csv", P_pi_sto_valid, delimiter=",")

    def get_V_pi_and_visualize(r_pi, P_pi, policy_type="deterministic", calc_type="closed_form",):
        print("="*50)
        env = init_env()
        if calc_type == "closed_form":
            V_pi = closed_form_solution(r_pi, P_pi, GAMMA)
        else:
            V_pi = iterative_policy_evaluation(r_pi, P_pi, GAMMA)

        print(f"V_pi ({policy_type}, {calc_type}):\n", V_pi)

        env.add_state_values(V_pi, precision=1)

        # add a title
        env.ax.set_title(f"State Value under {policy_type.capitalize()} Policy using {calc_type.replace('_', ' ').capitalize()} Method", fontsize=8, pad=40)

        env.render()
        env.canvas.savefig(f"A2/state_value_{policy_type}_{calc_type}.png")

        input(f"========== Now showing the state value for {policy_type} policy using {calc_type} method. Press Enter to continue... ==========\n")

        plt.close(env.canvas)
        env.canvas = None
        env.ax = None

        return V_pi

    get_V_pi_and_visualize(r_pi_det, P_pi_det, "deterministic", "closed_form")
    get_V_pi_and_visualize(r_pi_det, P_pi_det, "deterministic", "iterative")
    get_V_pi_and_visualize(r_pi_sto, P_pi_sto, "stochastic", "closed_form")
    get_V_pi_and_visualize(r_pi_sto, P_pi_sto, "stochastic", "iterative")


from A2.holiday_special import show_holiday_easter_egg
show_holiday_easter_egg()

