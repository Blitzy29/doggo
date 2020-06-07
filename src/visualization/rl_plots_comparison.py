import logging

import matplotlib.pyplot as plt
import numpy as np

from src.utils.maths import get_avg_n_points
from src.utils.plots import COLORS


logger = logging.getLogger(__name__)


def plot_comparison_evolution_reward(evo_training__evo_avg_reward_per_step, info_params):

    fig = plt.figure()

    cnt = 0
    for method in list(evo_training__evo_avg_reward_per_step.keys()):

        y = get_avg_n_points(evo_training__evo_avg_reward_per_step[method], n_points=100)
        x = range(len(y))

        plt.plot(
            x
            , y
            , label=method
            , marker='', color=COLORS[cnt], linewidth=1, alpha=0.75
        )
        cnt += 1

    plt.title('Evolution of Avg Reward \n per step per episode over time (smoothed)')
    plt.xlabel('Episode (percentile) \n ' + info_params)
    plt.ylabel('Avg Reward per step \n per episode (Smoothed)')
    plt.legend(bbox_to_anchor=(0.5, -0.10), loc="lower center",
               bbox_transform=fig.transFigure, ncol=4, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/figures/Comparison__reward.png', format='png', dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_comparison_evolution_steps(evo_training__evo_n_steps, nmax_steps, info_params):

    fig = plt.figure()

    cnt = 0
    for method in list(evo_training__evo_n_steps.keys()):

        y = get_avg_n_points(evo_training__evo_n_steps[method], n_points=100)
        x = range(len(y))

        plt.plot(
            x, y, label=method
            , marker='', color=COLORS[cnt], linewidth=1, alpha=0.75
        )
        cnt += 1

    plt.title('Episode Length over time (smoothed)')
    plt.axhline(nmax_steps, color='r')
    plt.axhline(0, color='b')
    plt.ylim([-10, nmax_steps*1.05])
    plt.xlabel('Episode (percentile) \n ' + info_params)
    plt.ylabel('Episode Length (Smoothed)')
    plt.legend(bbox_to_anchor=(0.5, -0.10), loc="lower center",
               bbox_transform=fig.transFigure, ncol=4, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/figures/Comparison__steps.png', format='png', dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close(fig)
