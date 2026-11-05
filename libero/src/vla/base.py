from typing import Dict, Optional, Tuple
import numpy as np


class InnerPolicy:
    """A frozen pretrained policy (VLA) that turns an observation + instruction
    into a chunk of low-level robot actions.

    Concrete implementations (e.g. `PIClient`) are built by hydra and handed to
    the env / outer policy through this interface, so nothing downstream needs to
    know whether the policy runs over HTTP, in-process, or on a real robot.
    """

    def sample_action(
        self,
        obs: Dict[str, np.ndarray],
        instruction: str,
        noise: Optional[np.ndarray] = None,
        meta: Optional[Dict] = None,
    ) -> np.ndarray:
        """Returns a (action_horizon, action_dim) chunk of low-level actions."""
        raise NotImplementedError

    def sample_actions(
        self,
        obs: Dict[str, np.ndarray],
        instruction: str,
        num_samples: int,
        noise: Optional[np.ndarray] = None,
        meta: Optional[Dict] = None,
    ) -> np.ndarray:
        """Returns multiple action chunks with shape ``(N, horizon, dim)``.

        Implementations with native model batching should override this method.
        The default keeps older inner policies compatible by issuing individual
        ``sample_action`` calls. When supplied, ``noise`` must have one leading
        sample dimension.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if noise is not None:
            noise = np.asarray(noise)
            if len(noise) != num_samples:
                raise ValueError(
                    f"noise has {len(noise)} samples, expected {num_samples}"
                )

        chunks = [
            self.sample_action(
                obs=obs,
                instruction=instruction,
                noise=None if noise is None else noise[index],
                meta=meta,
            )
            for index in range(num_samples)
        ]
        return np.stack(chunks)

    @property
    def noise_shape(self) -> Optional[Tuple[int, int]]:
        """Shape of the diffusion latent this policy accepts as `noise`.

        `(action_horizon, noise_dim)`, or None if the policy is not steerable by
        noise. Algorithms that act in the latent space (DSRL) read their action
        dim from here rather than duplicating it in the config.
        """
        return None

    @property
    def action_horizon(self) -> int:
        """Number of low-level actions returned by one `sample_action` call."""
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        """Dimensionality of the low-level action space."""
        raise NotImplementedError
