from PIL import Image
import numpy as np
from copy import deepcopy
import math
from typing import Dict, List

from src.envs.obswranglers.base import ObsWrangler


class LiberoObsWrangler(ObsWrangler):
    @staticmethod
    def quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """
        Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    @staticmethod
    def obs_to_qpos(obs: Dict[str, np.ndarray]) -> np.ndarray:
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                LiberoObsWrangler.quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
        return qpos

    @staticmethod
    def preprocess_rl_obs(
        obs: Dict[str, np.ndarray],
        cameras: List[str],
        use_state: bool,
        image_resolution: tuple = (224, 224),
    ) -> Dict[str, np.ndarray]:
        obs_copy = deepcopy(obs)

        # Process images
        imgs = [
            Image.fromarray(obs_copy[camera]).resize(image_resolution)
            for camera in cameras
        ]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        imgs = np.stack(imgs, axis=-1)  # (H, W, 3, num_cameras)

        state = [
            LiberoObsWrangler.obs_to_qpos(obs_copy)  # (q_pos_dim,)
        ]
        out = {
            "pixels": imgs,
            "state": np.concatenate(state),  # (state_dim,)
        }
        if not use_state:
            out.pop("state")
        return out
