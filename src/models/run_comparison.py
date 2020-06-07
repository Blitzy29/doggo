import json
import logging
import os

import dill

from src.utils.logger_module import setup_logging
from src.utils.plots import get_info_params
from src.visualization.plot_happiness import (
    plot_comparison_evolution_episode, plot_comparison_evolution_happiness)
from src.visualization.rl_plots_comparison import (
    plot_comparison_evolution_reward, plot_comparison_evolution_steps)

# python -m src.models.run_comparison

setup_logging(file_handler_name='run_comparison')

logger = logging.getLogger(__name__)


params = json.loads(open('src/models/run_comparison.json').read())


info_params_dict = {
    "nmax_steps": params['nmax_steps']
    , "gamma": params['gamma']
}
info_params = get_info_params(info_params_dict)

evo_training__evo_avg_reward_per_step = {}
evo_training__evo_n_steps = {}
evo_training__evo_avg_happiness = {}

evo_episode__happiness = {}

for method_id in params['list_method_ids']:
    logger.info(method_id)
    with open("models/{}__evo_training.pkl".format(method_id), "rb") as input_file:
        evo_training = dill.load(input_file)

    evo_training__evo_avg_reward_per_step[method_id] = evo_training['evo_avg_reward_per_step']
    evo_training__evo_n_steps[method_id] = evo_training['evo_n_steps']
    evo_training__evo_avg_happiness[method_id] = evo_training['evo_avg_happiness']

    with open("models/{}__evo_episode.pkl".format(method_id), "rb") as input_file:
        evo_episode = dill.load(input_file)

    evo_episode__happiness[method_id] = evo_episode['happiness']

logger.info('Reward')
plot_comparison_evolution_reward(evo_training__evo_avg_reward_per_step, info_params)
logger.info('Steps')
plot_comparison_evolution_steps(evo_training__evo_n_steps, info_params_dict['nmax_steps'], info_params)
logger.info('Happiness')
plot_comparison_evolution_happiness(evo_training__evo_avg_happiness, info_params)
logger.info('Episode')
plot_comparison_evolution_episode(evo_episode__happiness, info_params)

logger.info(os.path.basename(__file__)+' DONE')
