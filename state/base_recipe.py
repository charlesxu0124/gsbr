"""Load the matching flow-policy architecture."""

import json
from pathlib import Path
from agents.bc import get_config as bc_config


def get_config(path):
    config = bc_config()
    config.update(json.loads(Path(path).read_text())["base_policy"])
    return config
