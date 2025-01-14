import os

from external.tools.downloader import Downloader
from manifest import Manifest


def preload(manifest_file_key):
    known_keys = Manifest().get('files')
    if manifest_file_key not in known_keys:
        raise Exception('Requested file do not exists in manifest')
    manifest_file_val = known_keys[manifest_file_key]
    full_path = manifest_file_val['download_path']
    download_url = manifest_file_val['download_url']
    downloader = Downloader()
    downloader.download(download_url, full_path)
    return full_path
