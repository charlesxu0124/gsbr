import dataclasses
import gc
import glob
import os
import pickle
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from utils.flax_utils import atomic_pickle_dump


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = (
            False  # Whether to additionally return next actions; set outside the class.
        )

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample transitions, defaulting unlabelled dataset rows to offline data."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"])
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def sample_sequence(
        self,
        batch_size,
        sequence_length,
        discount,
        n_step=1,
        chunk_discount="step",
        idxs=None,
    ):
        """Sample action chunks with (n-step) chunked returns.

        Each sample is a window of ``W = n_step * H`` env steps starting at a
        random index whose first chunk of ``H = sequence_length`` steps lies
        inside one episode; the rest of the window may run past the episode end
        (or past the newest transition), after which ``valid`` is 0 and every
        quantity is frozen.  Rewards are accumulated once each with weight
        ``discount ** i`` (``chunk_discount="step"``, the per-env-step discount
        inside the chunk as well) or ``(discount ** H) ** (i // H)``
        (``chunk_discount="chunk"``: DICE-RL's undiscounted chunk sums with a
        per-chunk discount).  ``n_steps`` is the bootstrap discount exponent in
        env steps: the number of valid steps, rounded up to whole chunks for
        ``"chunk"``.  Agents bootstrap from ``next_observations[..., -1]`` with
        ``masks[..., -1]`` and ``discount ** n_steps``.

        Args:
            batch_size: Number of sequences to sample.
            sequence_length: Action-chunk length H.
            discount: Per-env-step discount factor.
            n_step: Chunks per return window (1 = plain chunk transitions).
            chunk_discount: "step" or "chunk", see above.
            idxs: Optional start indices (tests); sampled when None.

        Returns:
            dict with:
                observations:      (batch_size, obs_dim)  — initial obs at t
                actions:           (batch_size, H, action_dim)  — the first chunk
                next_observations: (batch_size, W, obs_dim)  — freezes at episode end
                rewards:           (batch_size, W)  — cumulative discounted rewards
                masks:             (batch_size, W)
                terminals:         (batch_size, W)
                valid:             (batch_size, W)  — 0 after the episode ends
                n_steps:           (batch_size,)  — bootstrap discount exponent
                is_offline:        (batch_size,)  — provenance of the initial transition
        """
        assert chunk_discount in ("step", "chunk"), chunk_discount
        window = sequence_length * n_step
        # Start indices are logical (0 = oldest transition).  On a full replay ring
        # the oldest row sits at ``pointer``, so logical indices are mapped to
        # storage rows below and a window never runs from the newest row into the
        # oldest data; on a plain dataset or an unfilled ring the two coincide.
        oldest = (
            self.pointer
            if isinstance(self, ReplayBuffer) and self.size == self.max_size
            else 0
        )
        if idxs is None:
            # Rejection sampling: sample random starts and reject any whose first
            # chunk contains a terminal before its last step.  O(batch_size * H)
            # per call with no state to invalidate.
            max_start = self.size - sequence_length
            if sequence_length == 1:
                idxs = np.random.randint(0, max_start + 1, size=batch_size)
            else:
                collected = []
                while len(collected) < batch_size:
                    n_try = max((batch_size - len(collected)) * 2 + 32, batch_size)
                    candidates = np.random.randint(0, max_start + 1, size=n_try)
                    # Shape: (n_try, H-1); reject if any terminal in [c, c+H-2].
                    offsets = np.arange(sequence_length - 1)
                    terminal_mat = self["terminals"][
                        (oldest + candidates[:, None] + offsets[None, :]) % self.size
                    ]
                    valid_mask = (terminal_mat == 0).all(axis=1)
                    collected.extend(candidates[valid_mask].tolist())
                idxs = np.array(collected[:batch_size])
        idxs = np.asarray(idxs)

        data = {k: v[(oldest + idxs) % self.size] for k, v in self.items()}

        rewards = np.zeros(data["rewards"].shape + (window,), dtype=float)
        masks = np.ones(data["masks"].shape + (window,), dtype=float)
        valid = np.ones(data["masks"].shape + (window,), dtype=float)
        next_observations = np.zeros(
            data["observations"].shape[:-1] + (window, data["observations"].shape[-1]),
            dtype=float,
        )
        actions = np.zeros(
            data["actions"].shape[:-1] + (sequence_length, data["actions"].shape[-1]),
            dtype=float,
        )
        terminals = np.zeros(data["terminals"].shape + (window,), dtype=float)

        for i in range(window):
            raw_idxs = idxs + i
            # Windows may run past the newest transition; those steps are invalid.
            in_range = (raw_idxs < self.size).astype(float)
            cur_idxs = (oldest + np.minimum(raw_idxs, self.size - 1)) % self.size
            if chunk_discount == "step":
                reward_weight = discount**i
            else:
                reward_weight = (discount**sequence_length) ** (i // sequence_length)

            if i == 0:
                rewards[..., 0] = self["rewards"][cur_idxs]
                masks[..., 0] = self["masks"][cur_idxs]
                terminals[..., 0] = self["terminals"][cur_idxs]
            else:
                valid[..., i] = (1.0 - terminals[..., i - 1]) * in_range
                rewards[..., i] = rewards[..., i - 1] + (
                    self["rewards"][cur_idxs] * reward_weight * valid[..., i]
                )
                masks[..., i] = np.minimum(
                    masks[..., i - 1], self["masks"][cur_idxs]
                ) * valid[..., i] + masks[..., i - 1] * (1.0 - valid[..., i])
                terminals[..., i] = np.maximum(
                    terminals[..., i - 1], self["terminals"][cur_idxs]
                )

            if i < sequence_length:
                actions[..., i, :] = self["actions"][cur_idxs]
            next_observations[..., i, :] = self["next_observations"][cur_idxs] * valid[
                ..., i : i + 1
            ] + next_observations[..., i - 1, :] * (1.0 - valid[..., i : i + 1])

        n_valid = valid.sum(axis=-1)
        if chunk_discount == "step":
            n_steps = n_valid
        else:
            n_steps = sequence_length * np.ceil(n_valid / sequence_length)

        return dict(
            observations=data["observations"].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            n_steps=n_steps,
            is_offline=data.get("is_offline", np.ones(idxs.shape, dtype=np.float32)),
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        # Plain datasets are demonstrations unless their rows explicitly carry
        # provenance (for example, a materialized replay buffer).
        result.setdefault("is_offline", np.ones(np.shape(idxs), dtype=np.float32))
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )

    def save(self, save_dir, epoch, prefix="dataset"):
        """Save the dataset to a file.

        Args:
            save_dir: Directory to save the dataset.
            epoch: Epoch number.
            prefix: Prefix for the filename (default: "dataset").
        """
        # Save only the valid portion of the dataset (up to size)
        valid_size = self.size
        dataset_data = {}
        for key, arr in self._dict.items():
            # Only save the valid portion
            dataset_data[key] = arr[:valid_size].copy()

        save_dict = dict(
            dataset_data=dataset_data,
            size=self.size,
            frame_stack=self.frame_stack,
            p_aug=self.p_aug,
            return_next_actions=self.return_next_actions,
            class_name=self.__class__.__name__,
        )
        save_path = os.path.join(save_dir, f"{prefix}_{epoch}.pkl")
        atomic_pickle_dump(save_dict, save_path)

        print(f"Saved {self.__class__.__name__} to {save_path} (size: {valid_size})")

    @classmethod
    def load(cls, save_path, *, legacy_is_offline=None):
        """Load a dataset from a file.

        Args:
            save_path: Path to the saved dataset file.
            legacy_is_offline: Known provenance for an old replay snapshot without
                an is_offline field: a scalar or one value per saved transition.
                Required for nonempty legacy replay, whose mixture cannot be
                inferred from transition values alone.

        Returns:
            Loaded Dataset or ReplayBuffer instance.
        """
        with open(save_path, "rb") as f:
            load_dict = pickle.load(f)

        # Determine the class to use
        class_name = load_dict.get("class_name", "Dataset")
        if class_name == "ReplayBuffer":
            dataset_class = ReplayBuffer
        else:
            dataset_class = cls

        # Get saved size and ensure data is truncated to this size
        saved_size = load_dict["size"]
        dataset_data = load_dict["dataset_data"]
        if class_name == "ReplayBuffer" and "is_offline" not in dataset_data:
            if legacy_is_offline is None and saved_size > 0:
                raise ValueError(
                    f"Replay snapshot {save_path} has no is_offline provenance. "
                    "Pass legacy_is_offline=0 only for known online-only replay, "
                    "or supply the known per-transition labels for mixed replay."
                )
            labels = 0.0 if legacy_is_offline is None else legacy_is_offline
            dataset_data["is_offline"] = np.broadcast_to(
                np.asarray(labels, dtype=np.float32), (saved_size,)
            ).copy()

        # Truncate arrays to saved size if they're longer (safety check)
        # This handles cases where saved arrays might be longer than the saved size
        truncated_data = {}
        for key, arr in dataset_data.items():
            if len(arr) > saved_size:
                truncated_data[key] = arr[:saved_size].copy()
            else:
                truncated_data[key] = arr

        # Create the dataset with truncated data
        dataset = dataset_class(truncated_data)

        # Restore attributes - override size with saved size
        dataset.size = saved_size
        dataset.frame_stack = load_dict.get("frame_stack")
        dataset.p_aug = load_dict.get("p_aug")
        dataset.return_next_actions = load_dict.get("return_next_actions", False)

        # Recompute terminal and initial locations
        dataset.terminal_locs = np.nonzero(dataset["terminals"] > 0)[0]
        dataset.initial_locs = np.concatenate([[0], dataset.terminal_locs[:-1] + 1])

        # Restore ReplayBuffer-specific attributes if applicable
        if isinstance(dataset, ReplayBuffer):
            saved_max_size = load_dict.get("max_size", saved_size)
            saved_pointer = load_dict.get("pointer", saved_size)
            # Re-expand arrays to max_size (save() trims to size for storage efficiency)
            if saved_max_size > saved_size:
                expanded_data = {}
                for key, arr in truncated_data.items():
                    full_arr = np.zeros(
                        (saved_max_size, *arr.shape[1:]), dtype=arr.dtype
                    )
                    full_arr[:saved_size] = arr
                    expanded_data[key] = full_arr
                dataset = dataset_class(expanded_data)
                dataset.size = saved_size
                dataset.frame_stack = load_dict.get("frame_stack")
                dataset.p_aug = load_dict.get("p_aug")
                dataset.return_next_actions = load_dict.get(
                    "return_next_actions", False
                )
                dataset.terminal_locs = np.nonzero(
                    expanded_data["terminals"][:saved_size] > 0
                )[0]
                dataset.initial_locs = np.concatenate(
                    [[0], dataset.terminal_locs[:-1] + 1]
                )
            dataset.max_size = saved_max_size
            dataset.pointer = saved_pointer % saved_max_size
            # Ensure size is correct (override any size computed from array length)
            dataset.size = saved_size

        print(f"Loaded {class_name} from {save_path} (size: {dataset.size})")
        return dataset


def process_offline_train_dataset(dataset, *, sparse: bool = False):
    """Materialize valid rows when needed, and optionally sparsify rewards.
    Args:
        dataset: ``Dataset`` or ``ReplayBuffer`` with ``size`` and transition keys.
        sparse: If True, reward is ``-1.0`` where dense reward was non-zero, else ``0.0``.
    Returns:
        A new ``Dataset`` (or the original ``Dataset`` if there is nothing to do).
    """
    if not sparse and not isinstance(dataset, ReplayBuffer):
        return dataset

    if sparse and not isinstance(dataset, ReplayBuffer):
        rewards = np.asarray(dataset["rewards"])
        sparse_rewards = np.where(rewards != 0.0, -1.0, 0.0).astype(rewards.dtype)
        return dataset.copy(add_or_replace=dict(rewards=sparse_rewards))

    # Preserve chronology when converting a replay ring to an ordinary dataset.
    n = int(dataset.size)
    if n == dataset.max_size and dataset.pointer != 0:
        idxs = (np.arange(n) + dataset.pointer) % n
        materialized = {k: v[idxs] for k, v in dataset.items()}
    else:
        materialized = {k: v[:n] for k, v in dataset.items()}
    if sparse:
        r = np.asarray(materialized["rewards"])
        materialized["rewards"] = np.where(r != 0.0, -1.0, 0.0).astype(r.dtype)

    ds = Dataset.create(**materialized)
    materialized = None
    gc.collect()
    return ds


def load_replay_buffer(save_dir, epoch, *, legacy_is_offline=None):
    """Load replay buffer from saved file.

    Args:
        save_dir: Directory containing the saved replay buffer.
        epoch: Epoch number.
        legacy_is_offline: Known scalar or per-row labels for legacy snapshots.

    Returns:
        Loaded ReplayBuffer instance.
    """
    replay_buffer_path = os.path.join(save_dir, f"replay_buffer_{epoch}.pkl")
    replay_buffer = Dataset.load(
        replay_buffer_path, legacy_is_offline=legacy_is_offline
    )
    replay_buffer.pointer = replay_buffer.pointer % replay_buffer.max_size
    return replay_buffer


def load_named_buffer(save_dir, epoch, prefix, *, legacy_is_offline=None):
    """Load a named replay buffer from a saved file.

    Args:
        save_dir: Directory containing the saved buffer.
        epoch: Epoch number.
        prefix: Filename prefix (e.g. 'success_buffer').
        legacy_is_offline: Known scalar or per-row labels for legacy snapshots.

    Returns:
        Loaded ReplayBuffer instance, or None if the file does not exist.
    """
    path = os.path.join(save_dir, f"{prefix}_{epoch}.pkl")
    if not os.path.exists(path):
        return None
    buf = Dataset.load(path, legacy_is_offline=legacy_is_offline)
    buf.pointer = buf.pointer % buf.max_size
    return buf


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset["terminals"] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config["value_p_curgoal"]
            + self.config["value_p_trajgoal"]
            + self.config["value_p_randomgoal"],
            1.0,
        )
        assert np.isclose(
            self.config["actor_p_curgoal"]
            + self.config["actor_p_trajgoal"]
            + self.config["actor_p_randomgoal"],
            1.0,
        )

        if self.config["frame_stack"] is not None:
            # Only support compact (observation-only) datasets.
            assert "next_observations" not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(
                    np.arange(self.size)
                )
                self.dataset = Dataset(
                    self.dataset.copy(dict(observations=stacked_observations))
                )

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )

        if "oracle_reps" in self.dataset:
            batch["value_goals"] = self.dataset["oracle_reps"][value_goal_idxs]
            batch["actor_goals"] = self.dataset["oracle_reps"][actor_goal_idxs]
        else:
            batch["value_goals"] = self.get_observations(value_goal_idxs)
            batch["actor_goals"] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)
        actor_successes = (idxs == actor_goal_idxs).astype(float)
        batch["actor_masks"] = 1.0 - actor_successes
        batch["actor_rewards"] = actor_successes - (
            1.0 if self.config["gc_negative"] else 0.0
        )

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                self.augment(
                    batch,
                    ["observations", "next_observations", "value_goals", "actor_goals"],
                )

        return batch

    def sample_goals(
        self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None
    ):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config["discount"]

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(
                p=1 - discount, size=batch_size
            )  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, final_state_idxs) * distances
                    + final_state_idxs * (1 - distances)
                )
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal),
                traj_goal_idxs,
                random_goal_idxs,
            )

            # Goals at the current state.
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs
            )

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: (
                    np.array(batched_random_crop(arr, crop_froms, padding))
                    if len(arr.shape) == 4
                    else arr
                ),
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config["frame_stack"] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(
                lambda arr: arr[idxs], self.dataset["observations"]
            )
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[
            np.searchsorted(self.initial_locs, idxs, side="right") - 1
        ]
        rets = []
        for i in reversed(range(self.config["frame_stack"])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(
                jax.tree_util.tree_map(
                    lambda arr: arr[cur_idxs], self.dataset["observations"]
                )
            )
        return jax.tree_util.tree_map(
            lambda *args: np.concatenate(args, axis=-1), *rets
        )


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support hierarchical goal-conditioned RL. It reads the following additional key from
    the config:
    - subgoal_steps (optional: value_subgoal_steps, actor_subgoal_steps): Subgoal steps. It is also possible to specify
        `value_subgoal_steps` and `actor_subgoal_steps` separately.
    - low_discount: If specified, return low-level value goals as well.
    """

    def compute_high_next_idxs(
        self, idxs, final_state_idxs, high_goal_idxs, subgoal_steps
    ):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        subgoal_steps = np.full(batch_size, subgoal_steps)

        # Clip to the end of the trajectory.
        subgoal_steps = np.minimum(subgoal_steps, final_state_idxs - idxs)

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < subgoal_steps)
        subgoal_steps = np.where(should_clip, diff_idxs, subgoal_steps)

        return idxs + subgoal_steps, subgoal_steps

    def get_high_actions(self, target_idxs, cur_idxs):
        if "oracle_reps" in self.dataset:
            return self.dataset["oracle_reps"][target_idxs]
        else:
            return self.get_observations(target_idxs)

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config["frame_stack"] is not None:
            batch["observations"] = self.get_observations(idxs)
            batch["next_observations"] = self.get_observations(idxs + 1)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config["value_p_curgoal"],
            self.config["value_p_trajgoal"],
            self.config["value_p_randomgoal"],
            self.config["value_geom_sample"],
        )
        value_subgoal_steps = (
            self.config["subgoal_steps"]
            if self.config.get("value_subgoal_steps") is None
            else self.config["value_subgoal_steps"]
        )
        high_value_next_idxs, high_value_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_subgoal_steps,
        )

        if "oracle_reps" in self.dataset:
            batch["high_value_reps"] = self.dataset["oracle_reps"][idxs]
            batch["high_value_goals"] = self.dataset["oracle_reps"][
                high_value_goal_idxs
            ]
            batch["high_value_actions"] = self.get_high_actions(
                high_value_next_idxs, idxs
            )
            batch["high_value_next_observations"] = self.get_observations(
                high_value_next_idxs
            )
        else:
            batch["high_value_reps"] = batch["observations"]
            batch["high_value_goals"] = self.get_observations(high_value_goal_idxs)
            batch["high_value_actions"] = self.get_high_actions(
                high_value_next_idxs, idxs
            )
            batch["high_value_next_observations"] = self.get_observations(
                high_value_next_idxs
            )
        batch["high_value_offsets"] = high_value_goal_idxs - idxs

        high_value_successes = (high_value_subgoal_steps < value_subgoal_steps).astype(
            float
        )
        batch["high_value_subgoal_steps"] = high_value_subgoal_steps
        batch["high_value_masks"] = 1.0 - high_value_successes
        if self.config["gc_negative"]:
            batch["high_value_rewards"] = -(
                1 - self.config["discount"] ** high_value_subgoal_steps
            ) / (1 - self.config["discount"])
        else:
            batch["high_value_rewards"] = (
                self.config["discount"] ** high_value_subgoal_steps
            ) * high_value_successes

        # Sample low-level value goals (if requested).
        if "low_discount" in self.config:
            low_value_goal_idxs = self.sample_goals(
                idxs,
                self.config["value_p_curgoal"],
                self.config["value_p_trajgoal"],
                self.config["value_p_randomgoal"],
                geom_sample=True,
                discount=self.config["low_discount"],
            )

            if "oracle_reps" in self.dataset:
                batch["low_value_goals"] = self.dataset["oracle_reps"][
                    low_value_goal_idxs
                ]
            else:
                batch["low_value_goals"] = self.get_observations(low_value_goal_idxs)
            successes = (idxs == low_value_goal_idxs).astype(float)
            batch["low_value_masks"] = 1.0 - successes
            batch["low_value_rewards"] = successes - (
                1.0 if self.config["gc_negative"] else 0.0
            )

        # One-step information.
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config["gc_negative"] else 0.0)

        # Sample high-level actor goals.
        high_actor_goal_idxs = self.sample_goals(
            idxs,
            self.config["actor_p_curgoal"],
            self.config["actor_p_trajgoal"],
            self.config["actor_p_randomgoal"],
            self.config["actor_geom_sample"],
        )
        actor_subgoal_steps = (
            self.config["subgoal_steps"]
            if self.config.get("actor_subgoal_steps") is None
            else self.config["actor_subgoal_steps"]
        )
        high_actor_next_idxs, high_actor_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_actor_goal_idxs,
            actor_subgoal_steps,
        )

        if "oracle_reps" in self.dataset:
            batch["high_actor_goals"] = self.dataset["oracle_reps"][
                high_actor_goal_idxs
            ]
            batch["high_actor_actions"] = self.get_high_actions(
                high_actor_next_idxs, idxs
            )
            batch["high_actor_next_observations"] = self.get_observations(
                high_actor_next_idxs
            )
        else:
            batch["high_actor_goals"] = self.get_observations(high_actor_goal_idxs)
            batch["high_actor_actions"] = self.get_high_actions(
                high_actor_next_idxs, idxs
            )
            batch["high_actor_next_observations"] = self.get_observations(
                high_actor_next_idxs
            )

        # Compute low-level actor goals.
        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)

        batch["low_actor_goals"] = self.get_high_actions(low_actor_goal_idxs, idxs)

        if self.config["p_aug"] is not None and not evaluation:
            if np.random.rand() < self.config["p_aug"]:
                raise NotImplementedError

        return batch


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Initial rows retain their provenance, defaulting to offline data. Later
        transitions default to online data when added without an explicit label.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        init_dataset = dict(init_dataset)
        init_size = get_size(init_dataset)
        if "is_offline" not in init_dataset:
            init_dataset["is_offline"] = np.ones(init_size, dtype=np.float32)
        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = min(init_size, size)
        dataset.pointer = init_size % size
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        if "is_offline" not in self._dict:
            # New replay starts online; load() requires explicit provenance for
            # legacy snapshots before constructing a populated buffer.
            self._dict["is_offline"] = np.zeros(self.max_size, dtype=np.float32)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        transition = dict(transition)
        transition.setdefault("is_offline", np.float32(0.0))
        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def truncate_episode(self):
        """End the latest episode before resetting an unrestored environment.

        Keep its mask and next observation so a reset boundary still bootstraps
        from the saved endpoint, unless it was already a true terminal.
        """
        if self.size:
            self["terminals"][(self.pointer - 1) % self.max_size] = 1.0

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

    def save(self, save_dir, epoch, prefix="replay_buffer"):
        """Save the replay buffer to a file.

        Args:
            save_dir: Directory to save the buffer.
            epoch: Epoch number.
            prefix: Prefix for the filename (default: "replay_buffer").
        """
        # Extract only the valid portion of the buffer (up to size)
        valid_size = self.size
        buffer_data = {}
        for key, arr in self._dict.items():
            # Only save the valid portion
            buffer_data[key] = arr[:valid_size].copy()

        save_dict = dict(
            dataset_data=buffer_data,  # Use same key as Dataset for compatibility
            size=self.size,
            pointer=self.pointer,
            max_size=self.max_size,
            frame_stack=self.frame_stack,
            p_aug=self.p_aug,
            return_next_actions=getattr(self, "return_next_actions", False),
            class_name=self.__class__.__name__,
        )
        save_path = os.path.join(save_dir, f"{prefix}_{epoch}.pkl")
        atomic_pickle_dump(save_dict, save_path)

        # Delete old copies only after the new file is fully written.
        for old in glob.glob(os.path.join(save_dir, f"{prefix}_*.pkl")):
            if old != save_path:
                os.remove(old)

        print(f"Saved replay buffer to {save_path} (size: {valid_size})")
