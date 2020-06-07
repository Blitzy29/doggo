import argparse
import json
import logging
import os

import dill

from src.utils.logger_module import setup_logging
from src.utils.plots import get_info_params
from src.visualization.plot_happiness import (make_gif_evolution_episode,
                                              plot_evolution_episode)

# python -m src.models.run_evolution_episode value_based/sarsa/config/sarsa_test

parser = argparse.ArgumentParser()
parser.add_argument("method_id_path", help="path towards the id parameters configuration to test")
args = parser.parse_args()
method_id = args.method_id_path.split('/')[-1]

setup_logging(file_handler_name='run_evolution_episode__'+method_id)

logger = logging.getLogger(__name__)


params = json.loads(open('src/models/run_evolution_episode.json').read())
params_method = json.loads(open('src/models/{}.json'.format(args.method_id_path)).read())

info_params_dict = {
    "nmax_steps": params['nmax_steps']
    , "gamma": params['gamma']
}
info_params = get_info_params(info_params_dict)

evo_episodes = []

# for episode in range(-1, params['n_episodes'], 1000)[1:]:
for episode in range(params['n_episodes']):
    try:
        with open("data/interim/{}__evo_episode_{}.pkl".format(method_id, episode), "rb") as input_file:
            evo_episode = dill.load(input_file)
            evo_episodes.append({'episode': episode, 'evo_episode': evo_episode['happiness']})
        logger.debug('episode :: {} :: {}'.format(episode, "{}__evo_episode_{}.pkl".format(method_id, episode)))
    except:
        pass


logger.info('Plot Evolution Episode')
plot_evolution_episode(evo_episodes, params['nmax_steps'], info_params)

logger.info('Make GIF evolution episode')
make_gif_evolution_episode()

logger.info(os.path.basename(__file__)+' DONE')
