"""GSBR collection and training with frozen pi0.5."""

from src.outer_policies.cached import CachedPolicy


class GSBRPolicy(CachedPolicy):
    def __init__(self, learner, **kwargs):
        super().__init__(rlt=learner, agent="gsbr", **kwargs)
