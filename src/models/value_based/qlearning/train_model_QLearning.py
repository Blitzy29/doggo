import argparse
import json
import logging
import os

import numpy as np
from pid import PidFile
from tqdm import tqdm

from src.data.env_dog import (env_reset, env_step, get_env_actions,
                              get_env_space, get_happiness)
from src.features.rl_tools import (define_training_convergence, epsilon_greedy,
                                   get_alpha, get_epsilon, init_Q,
                                   init_steps_per_state, select_best_action,
                                   update_steps_per_state)
from src.features.rl_tools_during_training import (display_analyse_Q,
                                                   env_render_episode,
                                                   initialize_evo_episode,
                                                   initialize_evo_training,
                                                   initialize_real_episode,
                                                   launch_checking,
                                                   save_models,
                                                   update_evo_episode,
                                                   update_evo_training,
                                                   update_real_episode)

from src.models.run_one_episode import run_episode
from src.models.value_based.qlearning.rl_tools_qlearning import update_Q_Qlearning

from src.utils.formats import str2bool
from src.utils.logger_module import setup_logging
from src.utils.music import play_music
from src.utils.plots import define_info_params_dict, get_info_params
from src.visualization.plot_happiness import plot_evolution_happiness
from src.visualization.rl_plots_evolution import (plot_evolution_reward,
                                                  plot_evolution_steps)


def main(args, logger):

    method_id = args.method_id
    logger.info(method_id)

    logger.debug(args)

    # Parametrisation
    params = json.loads(open('src/models/value_based/qlearning/config/{}.json'.format(method_id)).read())
    params_episode = json.loads(open('src/models/run_one_episode.json').read())

    logger.debug(params)

    info_params_dict = define_info_params_dict(params, method_id)
    info_params = get_info_params(info_params_dict)

    # Initializing environment
    action_dict, n_actions = get_env_actions()
    _, n_states = get_env_space()

    # Initializing the Q-matrix
    Q = init_Q(n_actions, params)
    Q_saved = Q.copy()

    # Initializing steps_per_state (count number of times we have been to each state)
    steps_per_state = init_steps_per_state()

    # Visualisation
    if args.update_episode_division == 0:
        n_episodes_save = 1e10
    else:
        n_episodes_save = int(np.ceil(params['n_episodes'] / 100 * args.update_episode_division))
    logger.debug('n_episodes_save :: {}'.format(n_episodes_save))

    if args.run_episode == 0:
        n_episodes_run = 1e10
    else:
        n_episodes_run = int(np.ceil(params['n_episodes'] / 100 * args.run_episode))
    logger.debug('n_episodes_run :: {}'.format(n_episodes_run))

    evolution_real_episode = initialize_real_episode()

    # Initializing the training
    evo_training = initialize_evo_training()

    # Training

    # Starting the learning
    pbar = tqdm(total=params['n_episodes'])

    while (not evo_training['convergence']) & (evo_training['episode'] < params['n_episodes']):

        # Get episode
        evo_episode = initialize_evo_episode()

        state1 = env_reset(params['start_at_random'])
        evo_episode['episode_step_happiness'].append(get_happiness(state1))

        # Update parameters
        epsilon = get_epsilon(
            params_epsilon=params['epsilon'], episode=evo_training['episode'],
            steps_state=steps_per_state[state1['state_id']])
        evo_episode['evo_epsilon'].append(epsilon)

        alpha = get_alpha(
            params_alpha=params['alpha'], episode=evo_training['episode'],
            steps_state=steps_per_state[state1['state_id']])
        evo_episode['evo_alpha'].append(alpha)

        while (not evo_episode['done']) and (evo_episode['n_episode_steps'] < params['nmax_steps']):

            action1 = epsilon_greedy(Q, state1['state_id'], n_actions, epsilon)
            steps_per_state = update_steps_per_state(steps_per_state, state1['state_id'])

            # Getting the next state
            state2, reward1, evo_episode['done'], info = env_step(state1, action1)

            # Update parameters
            epsilon = get_epsilon(
                params_epsilon=params['epsilon'], episode=evo_training['episode'],
                steps_state=steps_per_state[state2['state_id']])

            # Choosing the next action
            action2 = select_best_action(Q[state2['state_id']])

            # Learning the Q-value
            alpha = get_alpha(
                params_alpha=params['alpha'], episode=evo_training['episode'],
                steps_state=steps_per_state[state1['state_id']])

            method_params = {'alpha': alpha, 'gamma': params['gamma']}
            Q = update_Q_Qlearning(Q, state1['state_id'], action1, reward1, state2['state_id'], action2, method_params)

            evo_episode = update_evo_episode(evo_episode, reward1, state2, epsilon, alpha)

            # Updating the respective values
            state1 = state2
            evo_episode['n_episode_steps'] += 1

        # At the end of learning process
        if args.render_episode:
            logger.debug(env_render_episode(evo_training['episode'], evo_episode, epsilon, alpha))

        evo_training = update_evo_training(evo_training, evo_episode)

        # Run a real episode
        info_episode = run_episode(
            episode_prep={
                'method_id': method_id,
                'params': params_episode,
                'params_method': params,
                'action_dict': action_dict,
                'Q': Q
            }, args={
                'episode': evo_training['episode'],
                'plot_episode': False,
                'save_episode': False
            }, logger=logger)

        evolution_real_episode = update_real_episode(evolution_real_episode, info_episode)

        # if (evo_training['episode'] + 1) % n_episodes_run == 0:
        # save_models({'Q': Q, 'evo_training': evo_training}, method_id, final=False)
        #     os.system("python -m src.models.run_one_episode value_based/qlearning/config/{} {}".format(
        #         method_id, evo_training['episode']))

        if (evo_training['episode'] + 1) % n_episodes_save == 0:

            save_models({'evo_training': evo_training}, method_id, final=False)

            # info_params_dict['n_episodes'] = '{}/{}'.format(evo_training['episode'] + 1, params['n_episodes'])
            # info_params = get_info_params(info_params_dict)

            # evo_training['checking'], Q_saved = launch_checking(
            #    evo_training['checking'], Q_saved, Q, method_id, info_params, final=False)
            # save_models({'Q': Q, 'evo_training': evo_training}, method_id, final=False)

            # plot_evolution_reward(evo_training, method_id, info_params, final=False)
            # plot_evolution_steps(evo_training, method_id, params['nmax_steps'], info_params, final=False)
            # plot_evolution_happiness(evo_training, method_id, info_params, final=False)

            # evo_training['convergence'] = define_training_convergence(evo_training['checking']['evo_KLdiv'][-1], params)

        pbar.update(1)

    pbar.close()

    save_models({'evolution_real_episode': evolution_real_episode}, method_id, final=True)

    info_params_dict['n_episodes'] = '{}/{}'.format(evo_training['episode'] + 1, params['n_episodes'])
    info_params = get_info_params(info_params_dict)

    logger.info('checking')
    _, _ = launch_checking(
        evo_training['checking'], Q_saved, Q, method_id, info_params, final=True)
    save_models({'Q': Q, 'evo_training': evo_training}, method_id, final=True)

    logger.info('reward')
    plot_evolution_reward(evo_training, method_id, info_params, final=True)
    logger.info('steps')
    plot_evolution_steps(evo_training, method_id, params['nmax_steps'], info_params, final=True)
    logger.info('happiness')
    plot_evolution_happiness(evo_training, method_id, info_params, final=True)

    logger.info('episode')
    os.system("python -m src.models.run_one_episode value_based/qlearning/config/{} {} --save_episode --plot_episode".format(
        method_id, 'final'))

    # display_analyse_Q(Q, method_id, info_params, final=True)


# python -m src.models.value_based.qlearning.train_model_QLearning qlearning_test --render_episode --run_episode 10 --update_episode_division 10 --play_music
# python -m src.models.value_based.qlearning.train_model_QLearning qlearning_test --update_episode_division 10

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("method_id", help="id of the parameters configuration to test")
    parser.add_argument("--render_episode",
                        type=str2bool, nargs='?', const=True, default=False,
                        help="render information about every episode (in log).")
    parser.add_argument("--run_episode",
                        type=int, default=0,
                        help="run an full-exploitation episode every X percent.")
    parser.add_argument("--update_episode_division",
                        type=int, default=1, choices=[0, 1, 5, 10, 20, 50],
                        help="save Q and plot intermediate results every X percent.")
    parser.add_argument("--play_music",
                        type=str2bool, nargs='?', const=True, default=False,
                        help="play the spam song when finished.")
    args = parser.parse_args()

    setup_logging(file_handler_name=args.method_id)
    logger = logging.getLogger(__name__)

    with PidFile(pidname=args.method_id, piddir='data/log/pid') as p:
        main(args, logger)

    if args.play_music:
        play_music()

    logger.info(os.path.basename(__file__) + ' DONE')















