from core.tools import utils
from core.tools.downloader import Downloader

utils.enable_logging()
downloader = Downloader()
downloader.download("https://www.dropbox.com/s/5zhudqpupg061of/mnist10k.zip?dl=1", './mnist10k.zip')
