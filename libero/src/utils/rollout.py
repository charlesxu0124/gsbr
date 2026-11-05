from copy import deepcopy
import numpy as np
from typing import List

from src.utils.generalized_action import GeneralizedAction


class Rollout:
    def __init__(self):
        self.observations: List[dict] = []
        self.actions: List[GeneralizedAction] = []
        self.rewards: List[float] = []
        self.next_observations: List[dict] = []
        self.terminateds: List[bool] = []
        self.truncateds: List[bool] = []
        self.infos: List[dict] = []
        self.is_success = None

    def _add_to_observations(self, observation):
        self.observations.append(deepcopy(observation))

    def _add_to_next_observations(self, next_observation):
        self.next_observations.append(deepcopy(next_observation))

    def add_step(
        self,
        observation: dict,
        action: GeneralizedAction,
        reward: float,
        next_observation: dict,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        assert isinstance(action, (GeneralizedAction, np.ndarray)), (
            f"Action must be an instance of GeneralizedAction or np.ndarray, got {type(action)}."
        )

        self._add_to_observations(observation)
        self._add_to_next_observations(next_observation)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.infos.append(deepcopy(info))
        self.actions.append(deepcopy(action))

        done = terminated or truncated
        if done:
            assert "is_success" in info, (
                "Info dictionary must contain 'is_success' key upon episode termination."
            )
            self.is_success = info["is_success"]

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminateds = []
        self.truncateds = []
        self.infos = []
        self.is_success = None

    def __len__(self):
        return len(self.observations)

    def save_to_disk(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load_from_disk(path: str) -> "Rollout":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        rollout = Rollout()
        rollout.observations = data["observations"]
        rollout.actions = data["actions"]
        rollout.rewards = data["rewards"]
        rollout.next_observations = data["next_observations"]
        rollout.terminateds = data["terminateds"]
        rollout.truncateds = data["truncateds"]
        rollout.infos = data["infos"]
        rollout.is_success = data["is_success"]
        return rollout
