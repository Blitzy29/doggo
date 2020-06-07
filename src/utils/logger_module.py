import datetime
import logging
import logging.config
import os

import yaml


def setup_logging(
        file_handler_name=None,
        default_path='logging.yaml',
        default_level=logging.INFO,
        env_key='LOG_CFG'
):
    """
    Setup logging configuration
    file_handler_name: if not None, log will be store in 'data/log/DATE__file_handler_name.log'
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        if file_handler_name:
            file_handler_name = 'data/log/' + datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S") + '__' + file_handler_name + '.log'
            config['handlers']['file_handler']['filename'] = file_handler_name
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
