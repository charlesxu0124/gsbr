import numpy as np
from typing import Dict, Optional


class GeneralizedAction:
    def __init__(
        self,
        instruction: Optional[str] = None,
        low_level_action_chunk: Optional[np.ndarray] = None,
        noise: Optional[np.ndarray] = None,
        info: Optional[Dict] = None,
    ):
        self.instruction = instruction
        self.low_level_action_chunk = low_level_action_chunk
        self.noise = noise
        self.info = info
