import json
import os
import glob


class Manifest:
    _instance = None
    _file_path = os.path.dirname(os.path.abspath(__file__))
    _data = {}

    def __new__(cls):
        """Override __new__ to ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(Manifest, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Search all subdirectories below the base path for valid config.json files and load them."""
        if self._file_path is None:
            raise ValueError("Base path must be set before initialization.")

        config_files = glob.glob(os.path.join(self._file_path, '**', 'config.json'), recursive=True)

        for config_file in config_files:
            with open(config_file, 'r') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, dict) and 'manifest' in data.get('_comment'):
                        self._merge_config(data)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {config_file}")

    def _merge_config(self, new_data):
        """Merge the configuration data into the existing data."""
        self._data.update(new_data)

    def __getattr__(self, name):
        """Retrieve the attribute from the merged configuration data."""
        if name in self._data:
            return self._data[name]
        else:
            raise AttributeError(f"'Manifest' object has no attribute '{name}'")

    def get(self, name, default=None):
        """Retrieve the attribute, return default if not found."""
        return self._data.get(name, default)
