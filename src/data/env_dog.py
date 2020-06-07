import json
import logging
import random

from src.utils.maths import round_down, round_up


logger = logging.getLogger(__name__)

# Parameters specific to the environment
DOG_CHARACTERISTICS = json.loads(open('src/data/dog_configuration/sheppard.json').read())


def get_env_actions():
    action_dict = {
        0: 'NO ACTION',
        1: 'WALKING',
        2: 'EATING',
        3: 'PLAYING'
    }
    n_actions = len(action_dict)

    return action_dict, n_actions


# Observation spaces
def get_env_space():
    n_states = (10 ** DOG_CHARACTERISTICS['decimals_state'] + 1) ** 3
    return None, n_states


# Reset

def get_state_id(dog_state):
    return '{:01.4f}_{:01.4f}_{:01.4f}_{}'.format(
        dog_state['food'], dog_state['fat'], dog_state['affection'], dog_state['can_action_be_taken'])


def env_reset(start_at_random=False):

    if not start_at_random:
        dog_state = {
            'food': 0.5,
            'fat': 0,
            'affection': 0.5,
            'last_action_taken': 0,
            'minutes_since_last_action': 0,
            'can_action_be_taken': True
        }
    else:
        dog_state = {
            'food': round_up(random.random(), decimals=DOG_CHARACTERISTICS['decimals_state']),
            'fat': round_down(random.random(), decimals=DOG_CHARACTERISTICS['decimals_state']),
            'affection': round_up(random.random(), decimals=DOG_CHARACTERISTICS['decimals_state']),
            'last_action_taken': 0,
            'minutes_since_last_action': 0,
            'can_action_be_taken': True
        }

    dog_state['state_id'] = get_state_id(dog_state)

    return dog_state


def _apply_decreasing_rate(value: float, rate: float) -> float:
    """
    Apply a decreasing rate to a value
    :param value: current value
    :param rate: per second
    :return: updated value
    """
    return value - (60 * rate)


def _converge(value: float, target: float, ratio: float) -> float:
    diff: float = (target - value) * ratio
    return value + diff


def _update_food(dog_state):
    updated_food = _apply_decreasing_rate(
        dog_state['food'], DOG_CHARACTERISTICS['food_consumption_rate'])
    return round_down(max(0.0, updated_food), decimals=DOG_CHARACTERISTICS['decimals_state'])


def _update_fat(dog_state):
    updated_fat = dog_state['fat']
    return updated_fat


def _update_affection(dog_state):
    updated_affection = _apply_decreasing_rate(
        dog_state['affection'], DOG_CHARACTERISTICS['affection_consumption_rate'])
    return round_down(max(0.0, updated_affection), decimals=DOG_CHARACTERISTICS['decimals_state'])


def _update_if_walking(dog_state):
    updated_fat = round_down(
        _converge(dog_state['fat'], 0.0, DOG_CHARACTERISTICS['walking_fat_converge_rate']),
        decimals=DOG_CHARACTERISTICS['decimals_state'])
    updated_affection = round_up(
        _converge(dog_state['affection'], 1.0, DOG_CHARACTERISTICS['walking_affection_converge_rate']),
        decimals=DOG_CHARACTERISTICS['decimals_state'])
    return updated_fat, updated_affection


def _update_if_feeding(dog_state):
    updated_food = round_up(
        min(dog_state['food'] + DOG_CHARACTERISTICS['eating_food_increase'], 1.0),
        decimals=DOG_CHARACTERISTICS['decimals_state'])
    updated_fat = round_up(
        min(dog_state['fat'] + DOG_CHARACTERISTICS['eating_fat_increase'], 1.0),
        decimals=DOG_CHARACTERISTICS['decimals_state'])
    return updated_food, updated_fat


def _update_if_playing(dog_state):
    updated_fat = round_down(
        _converge(dog_state['fat'], 0.0, DOG_CHARACTERISTICS['playing_fat_converge_rate']),
        decimals=DOG_CHARACTERISTICS['decimals_state'])
    updated_affection = round_up(
        _converge(dog_state['affection'], 1.0, DOG_CHARACTERISTICS['playing_affection_converge_rate']),
        decimals=DOG_CHARACTERISTICS['decimals_state'])
    return updated_fat, updated_affection


def get_happiness(dog_state):
    happiness = min(dog_state['food'], 1.0 - dog_state['fat'], dog_state['affection'])
    return happiness


def _update_done(dog_state):
    happiness = get_happiness(dog_state)
    return happiness <= 0.0


# state2, reward1, done, info = env.step(action1)
def env_step(state1, action):

    info = {
        'action_taken_while_not_possible': False,
        'cause_of_death': 'no_death'
    }

    state2 = state1.copy()
    reward_penalty = 0

    # Affect of time
    state2['food'] = _update_food(state2)
    state2['fat'] = _update_fat(state2)
    state2['affection'] = _update_affection(state2)
    state2['minutes_since_last_action'] += 1

    # Applying action
    if action != 0:
        if state2['can_action_be_taken']:
            reward_penalty += 0.1
            state2['can_action_be_taken'] = False
            state2['minutes_since_last_action'] = 0
            state2['last_action_taken'] = action
        else:
            reward_penalty += 0.5
            info['action_taken_while_not_possible'] = True

    # Affect of actions
    if ((state2['last_action_taken'] == 1) &
            (state2['minutes_since_last_action'] == DOG_CHARACTERISTICS['walking_time'])):
        state2['fat'], state2['affection'] = _update_if_walking(state2)
        state2['can_action_be_taken'] = True

    if ((state2['last_action_taken'] == 2) &
            (state2['minutes_since_last_action'] == DOG_CHARACTERISTICS['eating_time'])):
        state2['food'], state2['fat'] = _update_if_feeding(state2)
        state2['can_action_be_taken'] = True

    if ((state2['last_action_taken'] == 3) &
            (state2['minutes_since_last_action'] == DOG_CHARACTERISTICS['playing_time'])):
        state2['fat'], state2['affection'] = _update_if_playing(state2)
        state2['can_action_be_taken'] = True

    done = _update_done(state2)
    if done:
        reward = -10
        if state2['food'] == 0:
            info['cause_of_death'] = 'food'
        elif (1.0 - state2['fat']) == 0:
            info['cause_of_death'] = 'fat'
        else:
            info['cause_of_death'] = 'affection'
    else:
        reward = min(state2['food'], 1.0 - state2['fat'], state2['affection']) - reward_penalty

    state2['state_id'] = get_state_id(state2)

    return state2, reward, done, info


# Render

def env_render(dog_state):
    logger.info(dog_state)
