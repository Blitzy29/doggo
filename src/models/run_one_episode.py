import argparse
import json
import logging
import os

import dill
from tqdm import tqdm

from src.data.env_dog import (env_reset, env_step, get_env_actions,
                              get_happiness)
from src.features.rl_tools import select_best_action
from src.features.rl_tools_during_training import (get_directory_model,
                                                   save_models)
from src.utils.formats import str2bool
from src.utils.logger_module import setup_logging
from src.utils.plots import get_info_params
from src.visualization.plot_happiness import plot_episode_happiness


def run_episode_prep(args, logger):

    # Parametrisation
    method_id = args.method_id_path.split('/')[-1]
    logger.debug(method_id)

    params_episode = json.loads(open('src/models/run_one_episode.json').read())
    logger.debug(params_episode)

    params_method = json.loads(open('src/models/{}.json'.format(args.method_id_path)).read())
    logger.debug(params_method)

    action_dict, n_actions = get_env_actions()

    # Variables
    directory = get_directory_model(args.episode == 'final')
    logger.debug(directory)

    if params_method['method'] == 'Double_Q_Learning':
        logger.debug(directory + "{}__Q1.pkl".format(method_id))
        with open(directory + "{}__Q1.pkl".format(method_id), "rb") as input_file:
            Q1 = dill.load(input_file)
        logger.debug(directory + "{}__Q2.pkl".format(method_id))
        with open(directory + "{}__Q2.pkl".format(method_id), "rb") as input_file:
            Q2 = dill.load(input_file)
        Q = (Q1, Q2)
    else:
        logger.debug(directory + "{}__Q.pkl".format(method_id))
        with open(directory + "{}__Q.pkl".format(method_id), "rb") as input_file:
            Q = dill.load(input_file)

    return {
        'method_id': method_id,
        'params': params_episode,
        'params_method': params_method,
        'action_dict': action_dict,
        'Q': Q
    }


def run_episode(episode_prep, args, logger):

    method_id, params_episode, params_method, action_dict = \
        episode_prep['method_id'], episode_prep['params'], episode_prep['params_method'],\
        episode_prep['action_dict']

    if params_method['method'] == 'Double_Q_Learning':
        (Q1, Q2) = episode_prep['Q']
    else:
        Q = episode_prep['Q']

    evo_episode = {
        'n_episode_steps': 0
        , 'done': False
        , 'action': []
        , 'action_taken': []
        , 'reward': []
        , 'happiness': []
        , 'food': []
        , 'inv_fat': []
        , 'affection': []
    }

    # Start episode and get initial observation
    state = env_reset()
    evo_episode['happiness'].append(get_happiness(state))
    evo_episode['food'].append(state['food'])
    evo_episode['inv_fat'].append(1-state['fat'])
    evo_episode['affection'].append(state['affection'])

    # pbar = tqdm(total=params['nmax_steps'])

    while (not evo_episode['done']) and (evo_episode['n_episode_steps'] < params_episode['nmax_steps']):

        # Get an action
        if params_method['method'] == 'Double_Q_Learning':
            action = select_best_action(Q_state=Q1[state['state_id']] + Q2[state['state_id']])
        else:
            action = select_best_action(Q_state=Q[state['state_id']])
        evo_episode['action'].append(action)

        # Perform a step
        state, reward, evo_episode['done'], info = env_step(state, action)
        evo_episode['reward'].append(reward)
        evo_episode['happiness'].append(get_happiness(state))
        if info['action_taken_while_not_possible']:
            evo_episode['action_taken'].append(0)
        else:
            evo_episode['action_taken'].append(action)
        evo_episode['food'].append(state['food'])
        evo_episode['inv_fat'].append(1-state['fat'])
        evo_episode['affection'].append(state['affection'])

        # Update n_steps
        evo_episode['n_episode_steps'] += 1

        # pbar.update(1)

    # pbar.close()

    evo_episode['avg_reward'] = sum(evo_episode['reward']) / evo_episode['n_episode_steps']
    evo_episode['n_steps'] = evo_episode['n_episode_steps']
    evo_episode['avg_happiness'] = sum(evo_episode['happiness']) / evo_episode['n_episode_steps']

    info_params = get_info_params({
        'method': params_method['method']
        , 'method_id': method_id
        , 'Avg Reward': round(evo_episode['avg_reward'], 4)
        , 'N-Steps': '{}/{}'.format(evo_episode['n_steps'], params_episode['nmax_steps'])
        , 'Avg Happiness': round(evo_episode['avg_happiness'], 4)
    })

    name_episode = '{}__{}'.format(method_id, args['episode'])
    if args['plot_episode']:
        plot_episode_happiness(evo_episode, action_dict, name_episode, info_params)
    if args['save_episode']:
        save_models({'evo_episode': evo_episode}, name_episode, final=(args['episode'] == 'final'))

    logger.debug(evo_episode.keys())

    return {
        'avg_reward': evo_episode['avg_reward'],
        'sum_reward': sum(evo_episode['reward']),
        'n_steps': evo_episode['n_steps'],
        'avg_happiness': evo_episode['avg_happiness'],
        'sum_happiness': sum(evo_episode['happiness']),
        'n_actions': len([action_taken for action_taken in evo_episode['action_taken'] if action_taken > 0]),
        'cause_of_death': info['cause_of_death'],
        'avg_food': sum(evo_episode['food']) / evo_episode['n_episode_steps'],
        'sum_food': sum(evo_episode['food']),
        'avg_inv_fat': sum(evo_episode['inv_fat']) / evo_episode['n_episode_steps'],
        'sum_inv_fat': sum(evo_episode['inv_fat']),
        'avg_affection': sum(evo_episode['affection']) / evo_episode['n_episode_steps'],
        'sum_affection': sum(evo_episode['affection'])
    }





# python -m src.models.run_one_episode value_based/sarsa/config/sarsa_test 1000 --plot_episode

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("method_id_path", help="path towards the id parameters configuration to test")
    parser.add_argument("episode", help="number of episodes trained, or final")
    parser.add_argument("--plot_episode", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="plot a summary of the episode.")
    parser.add_argument("--save_episode", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="save evolution of the episode.")
    args = parser.parse_args()

    method_id = args.method_id_path.split('/')[-1]
    setup_logging(file_handler_name='run_one_episode__' + method_id)
    logger = logging.getLogger(__name__)

    episode_prep = run_episode_prep(args, logger)

    info_episode = run_episode(
        episode_prep=episode_prep,
        args={
            'episode': args.episode,
            'plot_episode': args.plot_episode,
            'save_episode': args.save_episode
        },
        logger=logger)
    logger.debug(info_episode)

    logger.info(os.path.basename(__file__) + ' DONE')
