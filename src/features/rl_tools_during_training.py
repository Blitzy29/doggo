import dill
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.data.env_dog import get_happiness
from src.utils.maths import get_avg_n_points
from src.utils.plots import COLORS


logger = logging.getLogger(__name__)


def get_directory_model(final):
    if final:
        return 'models/'
    else:
        return 'data/interim/'


def get_directory_figures(final):
    if final:
        return 'data/figures/'
    else:
        return 'data/interim/'


def initialize_evo_training():
    return {
        'evo_avg_reward_per_step': []
        , 'evo_min_reward_per_step': []
        , 'evo_max_reward_per_step': []
        , 'evo_n_steps': []
        , 'evo_avg_happiness': []
        , 'evo_min_happiness': []
        , 'evo_max_happiness': []
        , 'checking': get_default_checking()
        , 'convergence': False
        , 'episode': 0
    }


def get_default_checking():
    return {
        'evo_epsilon': []
        , 'evo_alpha': []
        , 'evo_avg_Q': []
        , 'evo_KLdiv': []
        , 'step_reward': []
        , 'step_happiness': []
    }


def update_evo_training(evo_training, evo_episode):

    evo_training['evo_avg_reward_per_step'].append(sum(evo_episode['episode_step_reward']) / evo_episode['n_episode_steps'])
    evo_training['evo_min_reward_per_step'].append(min(evo_episode['episode_step_reward']))
    evo_training['evo_max_reward_per_step'].append(max(evo_episode['episode_step_reward']))

    evo_training['evo_n_steps'].append(evo_episode['n_episode_steps'])

    evo_training['evo_avg_happiness'].append(sum(evo_episode['episode_step_happiness']) / evo_episode['n_episode_steps'])
    evo_training['evo_min_happiness'].append(min(evo_episode['episode_step_happiness']))
    evo_training['evo_max_happiness'].append(max(evo_episode['episode_step_happiness']))

    evo_training['checking']['evo_epsilon'].append(sum(evo_episode['evo_epsilon']) / evo_episode['n_episode_steps'])
    evo_training['checking']['evo_alpha'].append(sum(evo_episode['evo_alpha']) / evo_episode['n_episode_steps'])

    evo_training['episode'] += 1

    return evo_training


def initialize_evo_episode():

    return {
        'n_episode_steps': 0
        , 'done': False
        , 'episode_step_reward': []
        , 'episode_step_happiness': []
        , 'evo_epsilon': []
        , 'evo_alpha': []
        , 'steps_episode': []
    }


def update_evo_episode(evo_episode, reward1, state2, epsilon, alpha):

    evo_episode['episode_step_reward'].append(reward1)
    evo_episode['episode_step_happiness'].append(get_happiness(state2))

    evo_episode['evo_epsilon'].append(epsilon)
    evo_episode['evo_alpha'].append(alpha)

    return evo_episode


def initialize_real_episode():

    return {
        'avg_reward': [],
        'sum_reward': [],
        'n_steps': [],
        'avg_happiness': [],
        'sum_happiness': [],
        'n_actions': [],
        'cause_of_death': [],
        'avg_food': [],
        'sum_food': [],
        'avg_inv_fat': [],
        'sum_inv_fat': [],
        'avg_affection': [],
        'sum_affection': []
    }


def update_real_episode(evolution_real_episode, info_episode):

    evolution_real_episode['avg_reward'].append(round(info_episode['avg_reward'], 4))
    evolution_real_episode['sum_reward'].append(round(info_episode['sum_reward'], 4))
    evolution_real_episode['n_steps'].append(int(info_episode['n_steps']))
    evolution_real_episode['avg_happiness'].append(round(info_episode['avg_happiness'], 4))
    evolution_real_episode['sum_happiness'].append(round(info_episode['sum_happiness'], 4))
    evolution_real_episode['n_actions'].append(int(info_episode['n_actions']))
    evolution_real_episode['cause_of_death'].append(info_episode['cause_of_death'])
    evolution_real_episode['avg_food'].append(round(info_episode['avg_food'], 4))
    evolution_real_episode['sum_food'].append(round(info_episode['sum_food'], 4))
    evolution_real_episode['avg_inv_fat'].append(round(info_episode['avg_inv_fat'], 4))
    evolution_real_episode['sum_inv_fat'].append(round(info_episode['sum_inv_fat'], 4))
    evolution_real_episode['avg_affection'].append(round(info_episode['avg_affection'], 4))
    evolution_real_episode['sum_affection'].append(round(info_episode['sum_affection'], 4))

    return evolution_real_episode


def env_render_episode(episode, evo_episode, epsilon, alpha):
    return 'Episode {}, Avg Reward :{}, Steps: {}, Avg Happiness: {}, Epsilon: {}, Alpha: {}'.format(
        episode + 1,
        round(sum(evo_episode['episode_step_reward']) / evo_episode['n_episode_steps'], 4),
        evo_episode['n_episode_steps'],
        round(sum(evo_episode['episode_step_happiness']) / evo_episode['n_episode_steps'], 4),
        epsilon, alpha)


def launch_checking(evo_training_checking, Q_saved, Q, method_id, info_params, final=True):

    if not final:
        evo_training_checking['evo_avg_Q'].append(get_avg_Q(Q))
        plot_avg_Q(evo_training_checking['evo_avg_Q'], method_id, info_params, final)

        evo_training_checking['evo_KLdiv'].append(calculate_KLdiv(Q_saved, Q))
        Q_saved = Q.copy()
        plot_KLdiv(evo_training_checking['evo_KLdiv'], method_id, info_params, final)

    plot_epsilon(evo_training_checking['evo_epsilon'], method_id, info_params, final)
    plot_alpha(evo_training_checking['evo_alpha'], method_id, info_params, final)

    return evo_training_checking, Q_saved


# Avg Q

def flatten(l):
    return [item for sublist in l for item in sublist]


def get_avg_Q(Q):

    q_Q = flatten(list(Q.values()))

    if len(q_Q) == 0:
        return 0
    else:
        return np.mean(q_Q)


def plot_avg_Q(avg_Q, method_id, info_params, final=True):

    y = avg_Q
    x = range(len(y))

    fig = plt.figure()
    plt.plot(x, y)
    plt.title('Average Q-values over time')
    plt.xlabel('Episode batch \n ' + info_params)
    plt.ylabel('Average Q-values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__avg_Q.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


# Kullback-Leibler divergence

def calculate_KLdiv(Q_saved, Q):

    (p_Q, q_Q) = ([], [])
    for i in list(Q.keys()):
        p_Q += list(Q_saved[i])
        q_Q += list(Q[i])

    if len(p_Q) == 0:
        return 0

    p_Q_norm = p_Q / sum(p_Q)
    q_Q_norm = q_Q / sum(q_Q)

    p_Q_norm_pos = [max(1e-12, i_p_Q) for i_p_Q in p_Q_norm]
    q_Q_norm_pos = [max(1e-12, i_q_Q) for i_q_Q in q_Q_norm]

    return stats.entropy(p_Q_norm_pos, q_Q_norm_pos)


def plot_KLdiv(evo_KLdiv, method_id, info_params, final=True):

    y = evo_KLdiv
    x = range(len(y))

    fig = plt.figure()
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('Kullback-Leibler divergence')
    plt.xlabel('Episode batch \n ' + info_params)
    plt.ylabel('Kullback-Leibler divergence between new and old (log scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__kldiv.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


# Epsilon

def plot_epsilon(evo_epsilon, method_id, info_params, final=True):

    y = get_avg_n_points(evo_epsilon, n_points=100)
    x = range(len(y))

    fig = plt.figure()
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('Epsilon')
    plt.xlabel('Episode (percentile)  \n ' + info_params)
    plt.ylabel('Epsilon (log scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__epsilon.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


# Alpha

def plot_alpha(evo_alpha, method_id, info_params, final=True):

    y = get_avg_n_points(evo_alpha, n_points=100)
    x = range(len(y))

    fig = plt.figure()
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('Alpha')
    plt.xlabel('Episode (percentile)  \n ' + info_params)
    plt.ylabel('Alpha (log scale)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__alpha.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


# Visualize Q

def display_analyse_Q(Q, method_id, info_params, final=True):

    analyse_Q = pd.DataFrame(columns=[
        'state_id', 'food_id', 'fat_id', 'affection_id', 'action_possible',
        'no_action', 'walking', 'feeding', 'playing',
    ])

    for i in tqdm(Q.keys()):
        i_info = i.split('_')
        Q_info = Q[i]
        analyse_Q = analyse_Q.append({
            'state_id': i,
            'food_id': float(i_info[0]),
            'fat_id': float(i_info[1]),
            'affection_id': float(i_info[2]),
            'action_possible': i_info[3],
            'no_action': Q_info[0],
            'walking': Q_info[1],
            'feeding': Q_info[2],
            'playing': Q_info[3]
            }, ignore_index=True)

    for state_characteristic in ['food_id', 'fat_id', 'affection_id']:
        logger.info(state_characteristic)
        display_state_characteristic(analyse_Q, state_characteristic, 'True', method_id, info_params, final)
        display_state_characteristic(analyse_Q, state_characteristic, 'False', method_id, info_params, final)


def display_state_characteristic(analyse_Q, state_characteristic, action_possible, method_id, info_params, final=True):

    Q_groupedby_state = analyse_Q[analyse_Q['action_possible'] == action_possible].groupby(state_characteristic).agg({
        'no_action': 'mean',
        'walking': 'mean',
        'feeding': 'mean',
        'playing': 'mean'
    })
    Q_groupedby_state.reset_index(inplace=True)

    fig = plt.figure()

    cnt = 0
    for action in ['no_action', 'walking', 'feeding', 'playing']:
        plt.plot(
            Q_groupedby_state[state_characteristic], Q_groupedby_state[action], label=action
            , marker='', color=COLORS[cnt], linewidth=1, alpha=0.75
        )
        cnt += 1

    plt.title('Avg Q-Value for state characteristic / action')
    plt.xlabel(state_characteristic + ' \n ' + info_params)
    plt.ylabel('Avg Q-Value')
    plt.xlim([-0.05, 1.05])
    plt.legend(bbox_to_anchor=(0.5, -0.10), loc="lower center",
               bbox_transform=fig.transFigure, ncol=4, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__Q_value_{}_{}.png'.format(
        method_id, state_characteristic, action_possible),
                format='png', dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


# Saving models

def save_models(models_to_save, method_id, final=True):

    directory = get_directory_model(final) + '{}__{}.pkl'

    for key_args in models_to_save:
        with open(directory.format(method_id, key_args), 'wb') as file:
            dill.dump(models_to_save[key_args], file)
