import glob
import logging

import imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.utils.maths import get_avg_n_points
from src.utils.plots import COLORS


logger = logging.getLogger(__name__)


def get_directory_figures(final):
    if final:
        return 'data/figures/'
    else:
        return 'data/interim/'


def plot_evolution_happiness(evo_training, method_id, info_params, final=True):

    avg_happiness = get_avg_n_points(evo_training['evo_avg_happiness'], n_points=100)
    min_happiness = get_avg_n_points(evo_training['evo_min_happiness'], n_points=100)
    max_happiness = get_avg_n_points(evo_training['evo_max_happiness'], n_points=100)
    x = range(len(avg_happiness))

    fig = plt.figure()
    plt.plot(x, avg_happiness)
    plt.plot(x, min_happiness, linestyle=':', color='lightcoral')
    plt.plot(x, max_happiness, linestyle=':', color='lightgreen')
    plt.title('Min/Avg/Max Happiness over time (smoothed)')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Episode (percentile) \n ' + info_params)
    plt.ylabel('Min/Avg/Max Happiness (smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_directory_figures(final) + '{}__happiness.png'.format(method_id), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


def plot_episode_happiness(evo_episode, action_dict, method, info_params):

    inv_action_dict = {v: k for k, v in action_dict.items()}

    HAPPINESS_patch = mpatches.Patch(color=COLORS[0], label='Happiness')

    WALKING_patch = mpatches.Patch(color=COLORS[inv_action_dict['WALKING'] + 3], label='Walking')
    EATING_patch = mpatches.Patch(color=COLORS[inv_action_dict['EATING'] + 3], label='Eating')
    PLAYING_patch = mpatches.Patch(color=COLORS[inv_action_dict['PLAYING'] + 3], label='Playing')

    FOOD_patch = mpatches.Patch(color=COLORS[1], label='Food')
    INV_FAT_patch = mpatches.Patch(color=COLORS[2], label='Fat (inv.)')
    AFFECTION_patch = mpatches.Patch(color=COLORS[3], label='Affection')

    fig = plt.figure(figsize=(20, 5))

    y = evo_episode['happiness']
    x = range(len(y))
    plt.plot(x, y, label='happiness', color=COLORS[0], linewidth=2)

    cnt = 1
    for i_characteristic in ['food', 'inv_fat', 'affection']:
        y = evo_episode[i_characteristic]
        x = range(len(y))
        plt.plot(x, y, label=i_characteristic, linewidth=0.5, color=COLORS[cnt])
        cnt += 1

    for i in range(len(evo_episode['action'])):
        if evo_episode['action'][i] != 0:
            plt.axvline(x=i, color=COLORS[evo_episode['action'][i] + 3], linewidth=0.75, alpha=0.5,
                        label=action_dict[evo_episode['action'][i]])
        if evo_episode['action_taken'][i] != 0:
            plt.axvline(x=i, color=COLORS[evo_episode['action_taken'][i] + 3], linewidth=2, alpha=1,
                        label=action_dict[evo_episode['action_taken'][i]])

    plt.title('{} - Happiness over time'.format(method))
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Episode \n' + info_params)
    plt.ylabel('Happiness')
    plt.legend(handles=[HAPPINESS_patch, FOOD_patch, INV_FAT_patch, AFFECTION_patch,
                        WALKING_patch, EATING_patch, PLAYING_patch],
               loc="upper left", bbox_transform=fig.transFigure, ncol=4, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/figures/{}__episode_happiness.png'.format(method), format='png', dpi=500)
    # plt.show()
    plt.close(fig)


def plot_comparison_evolution_happiness(evo_training__evo_avg_happiness, info_params):

    fig = plt.figure()

    cnt = 0
    for method in list(evo_training__evo_avg_happiness.keys()):

        y = get_avg_n_points(evo_training__evo_avg_happiness[method], n_points=100)
        x = range(len(y))

        plt.plot(
            x, y, label=method
            , marker='', color=COLORS[cnt], linewidth=1, alpha=0.75
        )
        cnt += 1

    plt.title('Happiness over time (smoothed)')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Episode (percentile) \n ' + info_params)
    plt.ylabel('Happiness (smoothed)')
    plt.legend(bbox_to_anchor=(0.5, -0.10), loc="lower center",
               bbox_transform=fig.transFigure, ncol=4, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/figures/Comparison__happiness.png', format='png', dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_comparison_evolution_episode(evo_episode__happiness, info_params):

    fig = plt.figure()

    cnt = 0
    for method_id in list(evo_episode__happiness.keys()):
        y = evo_episode__happiness[method_id]
        x = range(len(y))

        plt.plot(
            x, y, label=method_id
            , marker='', color=COLORS[cnt], linewidth=1, alpha=0.75
        )
        cnt += 1

    plt.title('Happiness over one episode')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Steps \n ' + info_params)
    plt.ylabel('Happiness (smoothed)')
    plt.legend(bbox_to_anchor=(0.5, -0.10), loc="lower center",
               bbox_transform=fig.transFigure, ncol=4, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/figures/Comparison__episode.png', format='png', dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def plot_evolution_episode(evo_episodes, nmax_steps, info_params):

    logger.info('init plot evolution episode :: ')

    for i in range(1, len(evo_episodes)+1):

        logger.debug('plot evolution episode :: {}'.format(i))

        evo_episodes_step = evo_episodes[:i]

        fig = plt.figure()

        cnt = 1
        for evo_episode in evo_episodes_step:
            y = evo_episode['evo_episode']
            x = range(len(y))

            plt.plot(
                x, y
                , marker='', color='b', linewidth=1, alpha=max(0.1, 1/(2**(len(evo_episodes_step)-cnt)))
            )
            cnt += 1

        plt.title('Happiness')
        plt.xlim([-10, nmax_steps])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Steps \n ' + info_params)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('data/figures/evolution_episode/Evolution_episode_{:04d}.png'.format(i),
                    format='png', dpi=1000, bbox_inches='tight')
        plt.close(fig)


def make_gif_evolution_episode():

    data_dir = 'data/figures/evolution_episode/'

    # List all images
    fileList = glob.glob(data_dir + '*.png')  # star grabs everything, can change to *.png for only png files
    fileList.sort()

    images = []
    for filename in fileList:
        images.append(imageio.imread(filename))
    imageio.mimsave(data_dir + 'evolution_episode.gif', images)
