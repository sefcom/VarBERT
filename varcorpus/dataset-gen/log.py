import logging.config
import os

def setup_logging(LOG_DIRECTORY, DEBUG=False):
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

    log_level = 'DEBUG' if DEBUG else 'INFO'
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {
            'default_formatter': {
                'format': '%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(lineno)d : %(message)s'
            }
        },

        'handlers': {
            'main_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'default_formatter',
                'filename': os.path.join(LOG_DIRECTORY, 'varcorpus.log')
            }
        },

        'loggers': {
            'main': {
                'handlers': ['main_handler'],
                'level': log_level,
                'propagate': False
            }
        }
    }

    logging.config.dictConfig(LOGGING_CONFIG)