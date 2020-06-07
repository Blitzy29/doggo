import logging
import random
from collections import defaultdict

import numpy as np


logger = logging.getLogger(__name__)


# Initialising Q

def init_Q(n_actions, params):
    """
    @param n_actions the number of actions
    @param params parameters of the all training
    """
    logger.debug('init_Q_type :: {}'.format(params['init_Q_type']))
    if params['init_Q_type'] == "ones":
        default_Q_values = np.ones(n_actions)
    elif params['init_Q_type'] == "random":
        default_Q_values = np.random.random(n_actions)
    elif params['init_Q_type'] == "zeros":
        default_Q_values = np.zeros(n_actions)
    elif params['init_Q_type'] == "optimum":
        V_max = get_discounted_reward(t=0, l_rewards_episode=np.ones(params['nmax_steps']), gamma=params['gamma'])
        default_Q_values = np.full(shape=n_actions, fill_value=float(V_max))
    else:
        default_Q_values = np.full(shape=n_actions, fill_value=float(params['init_Q_type']))

    def get_default_Q_values():
        return default_Q_values

    return defaultdict(get_default_Q_values)


# Initialising N

def init_N(n_actions):
    """
    @param n_actions the number of actions
    """
    default_N_values = np.zeros(n_actions)

    def get_default_N_values():
        return default_N_values

    return defaultdict(get_default_N_values)


# Initialising steps_per_state

def init_steps_per_state():
    """
    """
    return defaultdict(int)


# Choose an action

# Numpy generator
rng = np.random.default_rng()  # Create a default Generator.


def select_best_action(Q_state):
    winner = np.argwhere(Q_state == np.amax(Q_state))
    winner_list = winner.flatten().tolist()
    action = random.choice(winner_list)
    return action


# $\epsilon$-Greedy

def epsilon_greedy(Q, state_id, n_actions, epsilon, Q2=None):
    """
    @param Q Q values {state, action} -> value
    @param state_id state at current time
    @param n_actions number of actions
    @param epsilon for exploration
    @param Q2 Q-values {state, action} -> value used in Double-Q-Learning
    """
    if rng.uniform(0, 1) < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        if not Q2:
            action = select_best_action(Q[state_id])
        else:
            action = select_best_action(Q[state_id] + Q2[state_id])

    return action


# Discounted reward

def get_discounted_reward(t, l_rewards_episode, gamma):
    l_discounted_reward_episode = [
        t_prime_reward*(gamma**t_prime) for (t_prime, t_prime_reward) in enumerate(l_rewards_episode[t:])]
    G_k_t = sum(l_discounted_reward_episode)
    return G_k_t


def add_discounted_reward(steps_episode, gamma):
    l_rewards_episode = [step_episode['reward'] for step_episode in steps_episode]
    for (t, step_episode) in enumerate(steps_episode):
        step_episode['discounted_reward'] = get_discounted_reward(t, l_rewards_episode, gamma)
    return steps_episode


# Updating parameters

# Epsilon $\epsilon$ - Exploration rate

def get_epsilon(params_epsilon, episode, steps_state):

    if params_epsilon['decay_epsilon'] == 'fixed':
        epsilon = params_epsilon['init_epsilon']
    elif params_epsilon['decay_epsilon'] == 'per_episode':
        epsilon = params_epsilon['init_epsilon'] / (episode + 1)
    elif params_epsilon['decay_epsilon'] == 'per_state':
        epsilon = params_epsilon['init_epsilon'] / (steps_state + 1)
    elif params_epsilon['decay_epsilon'] == 'rate':
        epsilon = params_epsilon['init_epsilon'] * 0.99 ** episode
    else:
        logger.error('Epsilon - Decay unknown - Set to initial value')
        epsilon = params_epsilon['init_alpha']

    return max(epsilon, params_epsilon['min_epsilon'])


# Alpha $\alpha$ - Learning rate

def get_alpha(params_alpha, episode, steps_state):

    if params_alpha['decay_alpha'] == 'fixed':
        alpha = params_alpha['init_alpha']
    elif params_alpha['decay_alpha'] == 'per_episode':
        alpha = params_alpha['init_alpha'] / (episode + 1)
    elif params_alpha['decay_alpha'] == 'per_state':
        alpha = params_alpha['init_alpha'] / (steps_state + 1)
    elif params_alpha['decay_alpha'] == 'rate':
        alpha = params_alpha['init_alpha'] * 0.99 ** episode
    else:
        logger.error('Alpha - Decay unknown - Set to initial value')
        alpha = params_alpha['init_alpha']

    return max(alpha, params_alpha['min_alpha'])


# Update Steps per State

def update_steps_per_state(steps_per_state, state_id):

    steps_per_state[state_id] += 1

    return steps_per_state


def define_training_convergence(last_KLdiv, params):
    return last_KLdiv < params['KLdiv_convergence']
