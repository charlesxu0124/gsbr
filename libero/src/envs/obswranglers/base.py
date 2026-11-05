from typing import Dict, List
import numpy as np


class ObsWrangler:
    """Converts an observation from the environment into a standardized format for RL policies."""

    @staticmethod
    def preprocess_rl_obs(
        obs: Dict,
        cameras: List[str],
        use_state: bool,
        image_resolution: tuple = (224, 224),
    ) -> Dict[str, np.ndarray]:
        """
        Preprocesses the observation for RL policies.
        returns: {
            "pixels": np.ndarray of shape (height, width, channels, num_cameras),
            "state": np.ndarray of shape (state_dim,) if use_state
        }
        """
        raise NotImplementedError
