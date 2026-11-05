"""Load a selected paper agent configuration through ml_collections."""

import importlib
import json
from pathlib import Path


def get_config(path):
    saved = json.loads(Path(path).read_text())["agent"]
    config = importlib.import_module("agents." + saved["agent_name"]).get_config()
    config.update(saved)
    return config
