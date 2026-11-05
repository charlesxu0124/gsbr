"""pi0.5 sampling: one observation prefill, independent noise suffixes.

The cache is local to one request. It is never reused across observations.
This targets the deployed OpenPI Gemma cache layout (layers, batch, tokens,
heads, head_dim); it saves prefix computation, not cache storage during decode.
"""

import types

import jax
import jax.numpy as jnp
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils


def sample_shared_prefix(self, rng, observation, *, noise, num_steps=10):
    """Return N action chunks for one observation and N explicit noise tensors."""
    del rng  # Explicit noise keeps the two implementations directly comparable.
    if not self.pi05 or observation.state.shape[0] != 1:
        raise ValueError("Shared-prefix sampling requires pi0.5 and one observation")
    if noise.ndim != 3 or noise.shape[1:] != (self.action_horizon, self.action_dim):
        raise ValueError("noise must have shape (samples, action_horizon, action_dim)")
    count = noise.shape[0]
    if count < 1:
        raise ValueError("At least one sample is required")
    observation = model_lib.preprocess_observation(None, observation, train=False)
    tokens, mask, ar_mask = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm(
        [tokens, None],
        mask=make_attn_mask(mask, ar_mask),
        positions=jnp.cumsum(mask, axis=1) - 1,
    )

    # Gemma scans transformer layers on axis 0; the batch axis is 1.
    def expand_cache(value):
        if value.ndim != 5 or value.shape[1] != 1:
            raise ValueError("Unsupported OpenPI KV cache layout")
        return jnp.broadcast_to(value, (value.shape[0], count, *value.shape[2:]))

    cache = jax.tree.map(expand_cache, cache)
    mask = jnp.broadcast_to(mask, (count, mask.shape[1]))
    # pi0.5 embeds state in the prefix. Its action expert uses only noise/time;
    # keep a correctly batched Observation for OpenPI's shape checks.
    observation = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (count, *x.shape[1:])),
        observation,
    )
    dt = -1.0 / num_steps

    def step(carry):
        actions, time = carry
        suffix, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
            observation,
            actions,
            jnp.broadcast_to(time, count),
        )
        prefix_mask = jnp.broadcast_to(
            mask[:, None, :], (count, suffix.shape[1], mask.shape[1])
        )
        full_mask = jnp.concatenate(
            [prefix_mask, make_attn_mask(suffix_mask, suffix_ar_mask)], axis=-1
        )
        positions = (
            jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        )
        (_, output), _ = self.PaliGemma.llm(
            [None, suffix],
            mask=full_mask,
            positions=positions,
            kv_cache=cache,
            adarms_cond=[None, cond],
        )
        velocity = self.action_out_proj(output[:, -self.action_horizon :])
        return actions + dt * velocity, time + dt

    return jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))[0]


def make_shared_prefix_sampler(model):
    return nnx_utils.module_jit(types.MethodType(sample_shared_prefix, model))
