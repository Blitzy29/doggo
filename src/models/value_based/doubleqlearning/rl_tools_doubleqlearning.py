import logging


logger = logging.getLogger(__name__)


# Update Q-matrix (state-action value function)

def update_Q_doubleQlearning(Q1, Q2, state1_id, action1, reward1, state2_id, action2, method_params):

    previous_Q_value_state1 = Q1[state1_id].copy()

    predict = Q1[state1_id][action1]

    target = reward1 + method_params['gamma'] * Q2[state2_id][action2]

    new_Q_value = Q1[state1_id][action1] + method_params['alpha'] * (target - predict)
    previous_Q_value_state1[action1] = new_Q_value

    Q1[state1_id] = previous_Q_value_state1

    return Q1
