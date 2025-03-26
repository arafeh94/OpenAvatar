import logging
import os
import pickle
from zipfile import ZipFile

import wget
from future.moves import sys

from core.tools.file_tools import validate_path


class Downloader:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def download(self, url, into, extract_zip=True):
        if os.path.exists(into):
            self.logger.info("File {} already exists. Skipping download.".format(into))
            return True
        return self._get(url, into, extract_zip)

    def _bar_progress(self, current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    def _file_exists(self, downloaded_file):
        return os.path.isfile(downloaded_file)

    def _get(self, url, into, unzip=True, raise_exception=True):
        try:
            self.logger.info(f'downloading file into {into}')
            validate_path(into)
            self._download(url, into)
            if unzip:
                self.logger.info('extracting...')
                self._extract(into)
            return True
        except Exception as e:
            self.logger.info(f'error while downloading the file {e}')
            if raise_exception:
                raise e
            return False

    def _extract(self, file):
        file_extension = os.path.splitext(file)[-1].lower()
        directory = os.path.dirname(file)
        try:
            if file_extension == ".7z":
                import py7zr
                self.logger.info("Detected 7z archive. Extracting...")
                with py7zr.SevenZipFile(file, mode='r') as archive:
                    archive.extractall(directory)
                    self.logger.info("7z extraction complete!")
            elif file_extension == ".zip":
                self.logger.info("Extracting ZIP archive...")
                with ZipFile(file, 'r') as zipObj:
                    zipObj.extractall(directory)
                    self.logger.info("ZIP extraction complete!")
            else:
                self.logger.error(f"Unknown extension: {file_extension}. Skipping")

        except Exception as e:
            self.logger.error(f"An error occurred during extraction: {e}")
            raise e

    def _download(self, url, full_path):
        if 'mega.nz' in url:
            self.logger.info('mega.nz detected, using mega downloader...')
            directory = "/".join(full_path.split('/')[0:-1]) + "/"
            self._mega_downloader(url, directory)
        else:
            self._wget_downloader(url, full_path)

    def _wget_downloader(self, url, into):
        wget.download(url, into, bar=self._bar_progress)

    def _mega_downloader(self, url, directory):
        try:
            from mega import Mega
            m = Mega().login()
            m.download_url(url, directory)
        except Exception as ignore:
            self.logger.error('mega do not exists, please sure you install the mega.py package to the project.'
                              'you can install it using: pip install mega.py')
            self.logger.error('if the problem persist, make sure you are using python <=3.10')
            pass
        return True
