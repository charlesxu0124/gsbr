import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


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

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(
            self.size, size=num_idxs
        )  # NOTE: UNSEEDED RANDOMNESS LEAK

    def sample(self, batch_size: int, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def sample_trivial_sequence(self, batch_size: int, idxs=None):
        """
        Sample a batch of trivial sequences (length 1) from the dataset.
        Samples in a way that is consistent with `sample_sequence`, but allows for arbitrary dict keys.
        """
        sample = self.sample(batch_size, idxs)

        def add_seq_dim(arr):
            """
            Arrays are (B, ...) and we want (B, 1, ...).
            """
            return np.expand_dims(arr, axis=1)

        # Add a sequence dim to all arrays except for `observations`
        old_observations = sample["observations"]
        sample = jax.tree_util.tree_map(add_seq_dim, sample)  # creates a copy
        sample["observations"] = (
            old_observations  # `sample_sequence` does not modify `observations`
        )

        # Add a dummy `valid` key
        sample["valid"] = np.ones(sample["masks"].shape, dtype=float)  # (B, 1) float

        return sample

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        # NOTE: UNSEEDED RANDOMNESS LEAK
        data = {k: v[idxs] for k, v in self.items()}

        rewards = np.zeros(data["rewards"].shape + (sequence_length,), dtype=float)
        masks = np.ones(data["masks"].shape + (sequence_length,), dtype=float)
        valid = np.ones(data["masks"].shape + (sequence_length,), dtype=float)
        observations = np.zeros(
            data["observations"].shape[:-1]
            + (sequence_length, data["observations"].shape[-1]),
            dtype=float,
        )
        next_observations = np.zeros(
            data["observations"].shape[:-1]
            + (sequence_length, data["observations"].shape[-1]),
            dtype=float,
        )
        actions = np.zeros(
            data["actions"].shape[:-1] + (sequence_length, data["actions"].shape[-1]),
            dtype=float,
        )
        terminals = np.zeros(data["terminals"].shape + (sequence_length,), dtype=float)

        for i in range(sequence_length):
            cur_idxs = idxs + i

            if i == 0:
                rewards[..., 0] = self["rewards"][cur_idxs]
                masks[..., 0] = self["masks"][cur_idxs]
                terminals[..., 0] = self["terminals"][cur_idxs]
            else:
                valid[..., i] = 1.0 - terminals[..., i - 1]
                rewards[..., i] = rewards[..., i - 1] + (
                    self["rewards"][cur_idxs] * (discount**i) * valid[..., i]
                )
                masks[..., i] = np.minimum(
                    masks[..., i - 1], self["masks"][cur_idxs]
                ) * valid[..., i] + masks[..., i - 1] * (1.0 - valid[..., i])
                terminals[..., i] = np.maximum(
                    terminals[..., i - 1], self["terminals"][cur_idxs]
                )

            actions[..., i, :] = self["actions"][cur_idxs]
            next_observations[..., i, :] = self["next_observations"][cur_idxs] * valid[
                ..., i : i + 1
            ] + next_observations[..., i - 1, :] * (1.0 - valid[..., i : i + 1])
            observations[..., i, :] = self["observations"][cur_idxs]

        return dict(
            observations=data["observations"].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )

    def sample_sequence_dict_compatible(self, batch_size, sequence_length, discount):
        """
        Sample a sequence of transitions using the same logic as `sample_sequence`, but in a way that allows
        obs and actions to be a dict."""

        idxs = np.random.randint(
            self.size - sequence_length + 1, size=batch_size
        )  # NOTE: UNSEEDED RANDOMNESS LEAK
        data = {
            k: jax.tree_util.tree_map(lambda arr: arr[idxs], v) for k, v in self.items()
        }

        required_keys = [
            "observations",
            "actions",
            "rewards",
            "masks",
            "terminals",
            "next_observations",
        ]
        for key in required_keys:
            assert key in data, f"Key {key} is required for sampling sequences."

        def create_empty_with_seq_dim(arr):
            """(B, ...) -> (B, sequence_length, ...)"""
            return np.zeros(
                arr.shape[:1] + (sequence_length,) + arr.shape[1:], dtype=arr.dtype
            )

        batch = jax.tree_util.tree_map(
            create_empty_with_seq_dim,
            {
                k: data[k] for k in required_keys if k != "observations"
            },  # `observations` is handled separately
        )
        batch["valid"] = np.ones(
            batch["masks"].shape,
            dtype=float,  # 1 = valid, 0 = invalid
        )  # (B, sequence_length) float

        for i in range(sequence_length):
            cur_idxs = idxs + i  # For each idx, this points i steps into the future

            if i == 0:
                # batch['valid'][..., 0] # Untouched; first step is always valid
                batch["rewards"][..., 0] = self["rewards"][cur_idxs]
                batch["masks"][..., 0] = self["masks"][cur_idxs]
                batch["terminals"][..., 0] = self["terminals"][cur_idxs]
            else:
                batch["valid"][..., i] = (
                    1.0 - batch["terminals"][..., i - 1]
                )  # Valid if the previous step was not terminal
                batch["rewards"][..., i] = batch["rewards"][..., i - 1] + (
                    self["rewards"][cur_idxs] * (discount**i) * batch["valid"][..., i]
                )  # Sum of discounted, valid rewards
                # Masks:
                # - 1 if we should bootstrap, 0 if we should not (if valid)
                # - If not valid, copy the previous mask
                batch["masks"][..., i] = np.minimum(
                    batch["masks"][..., i - 1], self["masks"][cur_idxs]
                ) * batch["valid"][..., i] + batch["masks"][..., i - 1] * (
                    1.0 - batch["valid"][..., i]
                )
                batch["terminals"][..., i] = np.maximum(
                    batch["terminals"][..., i - 1], self["terminals"][cur_idxs]
                )  # Once terminated, always terminated

            def set_actions(dst, src):
                dst[:, i, ...] = src[cur_idxs]
                return dst

            def set_next_observations(dst, src):
                valid_i = batch["valid"][:, i].reshape(
                    (-1,) + (1,) * (dst.ndim - 2)
                )  # (B, 1, ..., 1) to broadcast over the leaf's trailing dims
                dst[:, i, ...] = src[cur_idxs] * valid_i + dst[:, i - 1, ...] * (
                    1.0 - valid_i
                )
                return dst

            batch["actions"] = jax.tree_util.tree_map(
                set_actions, batch["actions"], self["actions"]
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                set_next_observations,
                batch["next_observations"],
                self["next_observations"],
            )

        batch["observations"] = jax.tree_util.tree_map(
            lambda arr: arr.copy(), data["observations"]
        )
        return batch

    def sample_sequence_dict_compatible_optimized(
        self, batch_size, sequence_length, discount
    ):
        """
        Same semantics as `sample_sequence_dict_compatible`, but vectorized: exactly one
        fancy-index gather per key instead of a per-step Python loop, and no throwaway
        gathers of unused keys.
        """
        required_keys = [
            "observations",
            "actions",
            "rewards",
            "masks",
            "terminals",
            "next_observations",
        ]
        for key in required_keys:
            assert key in self._dict, f"Key {key} is required for sampling sequences."

        idxs = np.random.randint(
            self.size - sequence_length + 1, size=batch_size
        )  # NOTE: UNSEEDED RANDOMNESS LEAK
        all_idxs = idxs[:, None] + np.arange(
            sequence_length
        )  # (B, T), each row points 0..T-1 steps into the future

        rewards_raw = self["rewards"][all_idxs]  # (B, T)
        masks_raw = self["masks"][all_idxs]
        terminals_raw = self["terminals"][all_idxs]

        terminals = np.maximum.accumulate(
            terminals_raw, axis=1
        )  # Once terminated, always terminated
        valid = np.ones(
            (batch_size, sequence_length), dtype=float
        )  # 1 = valid, 0 = invalid
        valid[:, 1:] = 1.0 - terminals[:, :-1]  # Valid if no previous step was terminal

        # Sum of discounted, valid rewards
        discounts = discount ** np.arange(sequence_length)
        rewards = np.cumsum(rewards_raw * discounts * valid, axis=1).astype(
            rewards_raw.dtype
        )

        # Invalidity is absorbing, so frozen values are just the value at the last valid step
        last_valid = valid.sum(axis=1).astype(int) - 1  # (B,)
        clipped_steps = np.minimum(
            np.arange(sequence_length)[None, :], last_valid[:, None]
        )  # (B, T)

        # Running min of masks over valid steps, frozen once invalid
        masks = np.take_along_axis(
            np.minimum.accumulate(masks_raw, axis=1), clipped_steps, axis=1
        )

        clipped_idxs = (
            idxs[:, None] + clipped_steps
        )  # (B, T), held at the last valid step once invalid
        return dict(
            observations=jax.tree_util.tree_map(
                lambda arr: arr[idxs], self["observations"]
            ),
            actions=jax.tree_util.tree_map(lambda arr: arr[all_idxs], self["actions"]),
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=jax.tree_util.tree_map(
                lambda arr: arr[clipped_idxs], self["next_observations"]
            ),
        )

    def sample_sequence_dict_compatible_optimized_2(
        self, batch_size, sequence_length, discount
    ):
        """
        Same as `sample_sequence_dict_compatible_optimized`, but `next_observations` only contains
        the next obs at the last valid step, as a dummy time dim of length 1 (downstream code
        takes [-1] in the time dim, which is the only value the full sequence ever contributes).
        """
        required_keys = [
            "observations",
            "actions",
            "rewards",
            "masks",
            "terminals",
            "next_observations",
        ]
        for key in required_keys:
            assert key in self._dict, f"Key {key} is required for sampling sequences."

        idxs = np.random.randint(
            self.size - sequence_length + 1, size=batch_size
        )  # NOTE: UNSEEDED RANDOMNESS LEAK
        all_idxs = idxs[:, None] + np.arange(
            sequence_length
        )  # (B, T), each row points 0..T-1 steps into the future

        rewards_raw = self["rewards"][all_idxs]  # (B, T)
        masks_raw = self["masks"][all_idxs]
        terminals_raw = self["terminals"][all_idxs]

        terminals = np.maximum.accumulate(
            terminals_raw, axis=1
        )  # Once terminated, always terminated
        valid = np.ones(
            (batch_size, sequence_length), dtype=float
        )  # 1 = valid, 0 = invalid
        valid[:, 1:] = 1.0 - terminals[:, :-1]  # Valid if no previous step was terminal

        # Sum of discounted, valid rewards
        discounts = discount ** np.arange(sequence_length)
        rewards = np.cumsum(rewards_raw * discounts * valid, axis=1).astype(
            rewards_raw.dtype
        )

        # Invalidity is absorbing, so frozen values are just the value at the last valid step
        last_valid = valid.sum(axis=1).astype(int) - 1  # (B,)
        clipped_steps = np.minimum(
            np.arange(sequence_length)[None, :], last_valid[:, None]
        )  # (B, T)

        # Running min of masks over valid steps, frozen once invalid
        masks = np.take_along_axis(
            np.minimum.accumulate(masks_raw, axis=1), clipped_steps, axis=1
        )

        last_valid_idxs = (idxs + last_valid)[:, None]  # (B, 1)
        return dict(
            observations=jax.tree_util.tree_map(
                lambda arr: arr[idxs], self["observations"]
            ),
            actions=jax.tree_util.tree_map(lambda arr: arr[all_idxs], self["actions"]),
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=jax.tree_util.tree_map(
                lambda arr: arr[last_valid_idxs], self["next_observations"]
            ),
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        return result


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

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, pointer=0, size=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = size
        self.pointer = pointer

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

    def save(self, path):
        """Pickle the buffer arrays and write head to `path`."""
        from nicerl.utils.atomic_io import atomic_pickle

        atomic_pickle(
            dict(data=dict(self._dict), pointer=self.pointer, size=self.size), path
        )
        print(
            f"[ReplayBuffer] Saved to {path} (size={self.size}, pointer={self.pointer})"
        )

    @classmethod
    def load(cls, path):
        """Restore a buffer saved with `save` (capacity comes from the file)."""
        from nicerl.utils.atomic_io import load_pickle

        state = load_pickle(path)
        print(
            f"[ReplayBuffer] Loaded from {path} (size={state['size']}, pointer={state['pointer']})"
        )
        return cls(state["data"], pointer=state["pointer"], size=state["size"])
