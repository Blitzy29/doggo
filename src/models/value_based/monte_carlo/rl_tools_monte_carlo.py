import logging


logger = logging.getLogger(__name__)


# Update N-matrix

def update_N_MC(N, step_episode, method_monte_carlo, states_already_visited=[]):

    state_id = step_episode['state']['state_id']
    action = step_episode['action']

    previous_N_value_state = N[state_id].copy()

    if method_monte_carlo == 'first_visit':
        if state_id not in states_already_visited:
            new_N_value = N[state_id][action] + 1
            previous_N_value_state[action] = new_N_value

    if method_monte_carlo == 'every_visit':
        new_N_value = N[state_id][action] + 1
        previous_N_value_state[action] = new_N_value

    N[state_id] = previous_N_value_state

    return N


# Update Q-matrix (state-action value function)

def update_Q_MC(Q, N, step_episode, method_monte_carlo, states_already_visited=[]):

    state_id = step_episode['state']['state_id']
    action = step_episode['action']
    G_k_t = step_episode['discounted_reward']

    previous_Q_value_state = Q[state_id].copy()

    if method_monte_carlo == 'first_visit':
        if not state_id in states_already_visited:
            new_Q_value = Q[state_id][action] + (G_k_t - Q[state_id][action]) / N[state_id][action]
            previous_Q_value_state[action] = new_Q_value

    if method_monte_carlo == 'every_visit':
        new_Q_value = Q[state_id][action] + (G_k_t - Q[state_id][action]) / N[state_id][action]
        previous_Q_value_state[action] = new_Q_value

    Q[state_id] = previous_Q_value_state

    return Q
