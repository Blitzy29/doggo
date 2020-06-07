import numpy as np
import logging


logger = logging.getLogger(__name__)


# Update Q-matrix (state-action value function)
def update_Q_SARSA(Q, state1_id, action1, reward1, state2_id, action2, method_params, expected=False):

    previous_Q_value_state1 = Q[state1_id].copy()

    predict = Q[state1_id][action1]

    target = reward1 + method_params['gamma'] * Q[state2_id][action2]
    if expected:
        expected_value = np.mean(Q[state2_id])
        target = reward1 + method_params['gamma'] * expected_value

    new_Q_value = Q[state1_id][action1] + method_params['alpha'] * (target - predict)
    previous_Q_value_state1[action1] = new_Q_value

    Q[state1_id] = previous_Q_value_state1

    return Q
