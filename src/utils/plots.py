import logging

import matplotlib.colors as colors


logger = logging.getLogger(__name__)


def get_list_colors():
    colors_list = ['r','g','b','k','darkorange','y','lime','c','m'] + list(colors._colors_full_map.values())
    return colors_list


COLORS = get_list_colors()


def get_info_params(dict_args):

    info_params = ''
    cnt = 0
    for key_args in dict_args:
        if cnt == 3:
            info_params += ' \n '
            cnt = 0
        info_params += '{}: {} - '.format(key_args, dict_args[key_args])
        cnt += 1

    return info_params[:-3]


def define_info_params_dict(params, method_id):

    return {
        'method': params['method']
        , 'method_id': method_id
        , 'n_episodes': '{}/{}'.format(0, params['n_episodes'])
        , 'gamma': params['gamma']
        , 'nmax_steps': params['nmax_steps']
        , 'init_Q_type': params['init_Q_type']
        , 'start_at_random': params['start_at_random']
        , 'init_epsilon': params['epsilon']['init_epsilon']
        , 'decay_epsilon': params['epsilon']['decay_epsilon']
        , 'min_epsilon': params['epsilon']['min_epsilon']
        , 'init_alpha': params['alpha']['init_alpha']
        , 'decay_alpha': params['alpha']['decay_alpha']
        , 'min_alpha': params['alpha']['min_alpha']
        , 'KLdiv_convergence': params['KLdiv_convergence']
    }
