"""Shared ordered observation keys for Robomimic datasets and environments."""

SINGLE_ARM_OBSERVATION_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)
TRANSPORT_OBSERVATION_KEYS = (
    *SINGLE_ARM_OBSERVATION_KEYS,
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
)


def get_low_dim_obs_keys(env_name):
    """Return the same state-vector layout for offline data and live rollouts.

    Transport always includes both arms and uses the full 59D state.
    """
    if env_name.split("-", 1)[0] == "transport":
        return TRANSPORT_OBSERVATION_KEYS
    return SINGLE_ARM_OBSERVATION_KEYS
