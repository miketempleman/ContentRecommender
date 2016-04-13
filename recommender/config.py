import logging

LOGGER = None


def init_logging(log_level='INFO', logfile=None):
    global LOGGER

    log_format = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s')

    if logfile is None:
        logging.basicConfig(level=log_level, format=log_format)
    else:
        logging.basicConfig(level=log_level, format=log_format, filename=logfile)

    LOGGER = logging.getLogger('Recommender')
