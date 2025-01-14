import logging

from external.setup.phases.download_files import DownloadFilePhase
from external.tools import utils

utils.enable_logging()
phases = [DownloadFilePhase()]
logger = logging.getLogger('setup.py')
if __name__ == '__main__':
    for phase in phases:
        logger.info("Starting phase: {}".format(phase.__class__.__name__))
        logger.info("Phase Description: {}".format(phase.description()))
        phase.exec()
