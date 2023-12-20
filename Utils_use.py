import numpy as np
import matplotlib.pyplot as plt


def generate_rewards(num_states, each_step_reward, terminal_left_reward, terminal_right_reward):
    rewards = [each_step_reward] * num_states
    rewards[0] = terminal_left_reward
    rewards[-1] = terminal_right_reward

    return rewards


def generate_transition_prob(num_states, num_actions, misstep_prob=0):
    # 0 is left, 1 is right

    p = np.zeros((num_states, num_actions, num_states))

    for i in range(num_states):
        if i != 0:
            p[i, 0, i - 1] = 1 - misstep_prob
            p[i, 1, i - 1] = misstep_prob

        if i != num_states - 1:
            p[i, 1, i + 1] = 1 - misstep_prob
            p[i, 0, i + 1] = misstep_prob

    # Terminal States
    p[0] = np.zeros((num_actions, num_states))
    p[-1] = np.zeros((num_actions, num_states))

    return p