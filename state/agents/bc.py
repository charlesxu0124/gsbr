import math
from functools import partial
from typing import Any, Optional
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import TrainState
from utils.networks import MLP, ResidualMLP


def timestep_embedding(t, embedding_type="sinusoidal", emb_size=16, max_period=10000):
    """t is between [0, 1]."""
    t = jax.lax.convert_element_type(t, jnp.float32)
    if embedding_type == "fourier":
        freqs = jnp.arange(1, 33, dtype=t.dtype)
        angles = 2 * jnp.pi * t[:, None] * freqs[None]
        return jnp.concatenate([t[:, None], jnp.cos(angles), jnp.sin(angles)], axis=-1)
    if embedding_type != "sinusoidal":
        raise ValueError(f"Unknown timestep embedding: {embedding_type}")
    t = t * max_period
    dim = emb_size
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding


class FlowMatchingPolicy(nn.Module):
    """Flow matching policy for behavior cloning."""

    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)
    mlp_class: str = "mlp"
    time_embedding: str = "sinusoidal"

    @nn.compact
    def __call__(self, obs, noised_action, t):
        t_embedding = timestep_embedding(t, self.time_embedding)
        concat_input = jnp.concatenate([obs, noised_action, t_embedding], axis=-1)
        if self.mlp_class == "mlp":
            outputs = MLP(self.hidden_dims, activate_final=True, **self.mlp_kwargs)(
                concat_input
            )
            return nn.Dense(self.action_dim)(outputs)
        if self.mlp_class == "resmlp":
            return ResidualMLP(
                hidden_dims=self.hidden_dims,
                output_dim=self.action_dim,
                **self.mlp_kwargs,
            )(concat_input)
        raise ValueError(f"Unknown MLP class: {self.mlp_class}")


class BCAgent(flax.struct.PyTreeNode):
    """Flow matching behavior cloning agent."""

    rng: Any
    policy: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def _full_action_dim(self):
        return self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

    def policy_loss(self, batch, policy_params=None, rng=None):
        """Compute flow matching behavior cloning loss."""
        if rng is None:
            rng = self.rng
        if policy_params is None:
            policy_params = self.policy.params
        eps_rng, time_rng = jax.random.split(rng, 2)
        actions = batch["actions"]
        if self.config["action_chunking"]:
            actions = jnp.reshape(actions, (actions.shape[0], -1))
        x1 = actions
        x0 = jax.random.normal(eps_rng, x1.shape)
        if self.config["time_sampling"] == "uniform_discrete":
            t = (
                jax.random.randint(
                    time_rng, (x1.shape[0],), 0, self.config["denoise_steps"] + 1
                ).astype(jnp.float32)
                / self.config["denoise_steps"]
            )
        elif self.config["time_sampling"] == "beta":
            beta_a, beta_b = self.config["betas"]
            if beta_a == 1.0:
                uniform_t = jax.random.uniform(time_rng, (x1.shape[0],))
                t = 1 - (1 - uniform_t) ** (1 / beta_b)
            else:
                t = jax.random.beta(time_rng, beta_a, beta_b, (x1.shape[0],))
        else:
            raise ValueError(f"Unknown time sampling: {self.config['time_sampling']}")
        tv = t[..., None]
        x_t = x0 * (1 - tv) + x1 * tv
        vel = x1 - x0
        pred_vel = self.policy(batch["observations"], x_t, t, params=policy_params)
        bc_loss = jnp.mean((vel - pred_vel) ** 2)
        return (bc_loss, {"bc_loss": bc_loss})

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss for compatibility with main.py evaluation."""
        if rng is None:
            rng = self.rng
        policy_params = grad_params if grad_params is not None else self.policy.params
        loss, info = self.policy_loss(batch, policy_params=policy_params, rng=rng)
        return (loss, info)

    @jax.jit
    def update(self, batch):
        """Update the policy using flow matching behavior cloning."""
        new_rng, policy_rng = jax.random.split(self.rng)

        def policy_loss_fn(policy_params):
            return self.policy_loss(batch, policy_params, rng=policy_rng)

        new_policy, policy_info = self.policy.apply_loss_fn(loss_fn=policy_loss_fn)
        return (self.replace(rng=new_rng, policy=new_policy), policy_info)

    @partial(jax.jit, static_argnames=("denoise_steps",))
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any = None,
        temperature: float = 1.0,
        init_noise: Any = None,
        denoise_steps: Optional[int] = None,
    ) -> jnp.ndarray:
        """Sample actions from the flow matching policy.

        Args:
            observations: Observation batch or a single observation.
            seed: PRNG key used when no initial noise is provided.
            temperature: Kept for policy API compatibility.
            init_noise: Optional initial denoising noise, with shape
                ``(batch, full_action_dim)`` or ``(full_action_dim,)``.
            denoise_steps: Optional number of Euler integration steps. Defaults
                to the base policy's configured denoising steps.
        """
        denoise_steps = (
            self.config["denoise_steps"] if denoise_steps is None else denoise_steps
        )
        observations = observations[None] if observations.ndim == 1 else observations
        full_action_dim = self._full_action_dim()
        expected_noise_shape = (observations.shape[0], full_action_dim)
        if init_noise is None:
            assert seed is not None, (
                "`seed` is required when `init_noise` is not provided."
            )
            x = jax.random.normal(seed, expected_noise_shape)
        else:
            x = jnp.asarray(init_noise)
            x = x[None] if x.ndim == 1 else x
            assert x.shape == expected_noise_shape, (
                f"Initial noise shape {x.shape} does not match expected shape {expected_noise_shape}."
            )
        dt = 1.0 / denoise_steps

        def step(x, t):
            ti = jnp.ones((x.shape[0],)) * (t / denoise_steps)
            v = self.policy(observations, x, ti)
            x = x + v * dt
            return (x, None)

        actions, _ = jax.lax.scan(
            step,
            x,
            jnp.arange(denoise_steps),
            length=denoise_steps,
            unroll=self.config["denoise_unroll"],
        )
        actions = actions[0] if observations.shape[0] == 1 else actions
        action_clip = (
            jnp.full((self.config["action_dim"],), self.config["action_clip"])
            .at[-1]
            .set(self.config["gripper_action_clip"])
        )
        if self.config["action_chunking"]:
            action_clip = jnp.tile(action_clip, self.config["horizon_length"])
        actions = jnp.clip(actions, -action_clip, action_clip)
        return actions

    @classmethod
    def create(
        cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray, config
    ):
        """Create a new flow matching BC agent."""
        rng = jax.random.PRNGKey(seed)
        rng, policy_key = jax.random.split(rng)
        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config["action_dim"] = action_dim
        full_action_dim = (
            action_dim * config["horizon_length"]
            if config["action_chunking"]
            else action_dim
        )
        try:
            activation_fn = getattr(nn, config["activation"])
        except:
            activation_fn = nn.gelu
        policy_def = FlowMatchingPolicy(
            config["hidden_dims"],
            full_action_dim,
            mlp_class=config["mlp_class"],
            time_embedding=config["time_embedding"],
            mlp_kwargs=dict(
                activation=activation_fn, layer_norm=config["use_layer_norm"]
            ),
        )
        ex_full_actions = jnp.zeros((ex_actions.shape[0], full_action_dim))
        policy_params = policy_def.init(
            policy_key, ex_observations, ex_full_actions, jnp.zeros(ex_actions.shape[0])
        )["params"]
        policy = TrainState.create(
            policy_def, policy_params, tx=optax.adam(learning_rate=config["bc_lr"])
        )
        config_dict = flax.core.FrozenDict(**config)
        return cls(rng, policy=policy, config=config_dict)


def get_config():
    """Get default configuration for flow matching BC agent."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name="bc",
            batch_size=256,
            hidden_dims=(512, 512, 512, 512),
            bc_lr=0.0003,
            use_layer_norm=1,
            activation="gelu",
            denoise_steps=16,
            denoise_unroll=1,
            mlp_class="mlp",
            time_sampling="uniform_discrete",
            betas=(1.0, 1.5),
            time_embedding="sinusoidal",
            horizon_length=1,
            action_chunking=True,
            action_clip=1.0,
            gripper_action_clip=1.0,
        )
    )
    return config
