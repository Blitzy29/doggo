import logging

import matplotlib.pyplot as plt

from src.utils.maths import (get_avg_n_points, get_max_n_points,
                             get_min_n_points)


logger = logging.getLogger(__name__)


def get_directory_figures(final):
    if final:
        return 'data/figures/'
    else:
        return 'data/interim/'


# Reward

def plot_evolution_reward(evo_training, method_id, info_params, final=True):

    avg_reward = get_avg_n_points(evo_training['evo_avg_reward_per_step'], n_points=100)
    min_reward = get_avg_n_points(evo_training['evo_min_reward_per_step'], n_points=100)
    max_reward = get_avg_n_points(evo_training['evo_max_reward_per_step'], n_points=100)
    x = range(len(avg_reward))

    fig = plt.figure()
    plt.plot(x, avg_reward)
    plt.plot(x, min_reward, linestyle=':', color='lightcoral')
    plt.plot(x, max_reward, linestyle=':', color='lightgreen')
    plt.title('Evolution of Min/Avg/Max Reward \n per step per episode over time (smoothed)')
    plt.xlabel('Episode (percentile) \n ' + info_params)
    plt.ylabel('Min/Avg/Max Reward \n per step per episode (smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__reward.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


# Number of steps

def plot_evolution_steps(evo_training, method_id, nmax_steps, info_params, final=True):

    fig = plt.figure()

    avg_steps = get_avg_n_points(evo_training['evo_n_steps'], n_points=100)
    x = range(len(avg_steps))
    plt.plot(x, avg_steps)

    min_steps = get_min_n_points(evo_training['evo_n_steps'], n_points=min(100, len(evo_training['evo_n_steps'])))
    x = range(len(min_steps))
    plt.plot(x, min_steps, linestyle=':', color='lightcoral')

    max_steps = get_max_n_points(evo_training['evo_n_steps'], n_points=min(100, len(evo_training['evo_n_steps'])))
    x = range(len(max_steps))
    plt.plot(x, max_steps, linestyle=':', color='lightgreen')

    plt.title('Episode Length over time (smoothed)')
    plt.axhline(nmax_steps, color='r')
    plt.axhline(0, color='b')
    plt.ylim([-10, nmax_steps * 1.05])
    plt.xlabel('Episode (percentile) \n ' + info_params)
    plt.ylabel('Episode Length (smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__steps.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)
