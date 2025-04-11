import importlib
import logging


def get_class_instance(class_path: str, *args, **kwargs):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ref = getattr(module, class_name)
    return class_ref(*args, **kwargs)


def enable_logging(file_name=None, level=logging.INFO):
    if file_name:
        logging.basicConfig(filename=file_name, filemode='w', datefmt='%H:%M:%S', level=level)
    else:
        logging.basicConfig(level=level)
