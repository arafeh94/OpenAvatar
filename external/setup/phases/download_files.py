import logging

from external.setup.phase import SetupPhase
from external.tools.downloader import Downloader
from external.tools.preloader import preload
from manifest import Manifest


class DownloadFilePhase(SetupPhase):
    def exec(self):
        self.logger.info("Checking required files")
        files_desc = Manifest().get('files')
        required_files = []
        for file_key in files_desc:
            if 'required' in files_desc[file_key] and files_desc[file_key]['required']:
                self.logger.info("Adding manifest key [{}] to preloading queue".format(file_key))
                required_files.append(file_key)
        self.logger.info("Preloading {} files".format(len(required_files)))
        for key in required_files:
            preload(key)
        self.logger.info("Finished preloading")

    def description(self):
        return "Preloading required files for functionalities like avatar main objects"
