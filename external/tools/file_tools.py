import os


def validate_path(file_path):
    parent_dir = os.path.dirname(file_path)
    if len(parent_dir) and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
