"""RLT collection and training with frozen pi0.5."""

from src.outer_policies.cached import CachedPolicy


class RLTPolicy(CachedPolicy):
    def __init__(self, learner, **kwargs):
        super().__init__(rlt=learner, agent="rlt", **kwargs)
