import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from nicerl.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from nicerl.utils.networks import Value, LogParam, MLP, TanhNormal
from nicerl.utils.encoders.networks import (
    PixelStateEncoder,
    ObservationActionMultiplexer,
)

from functools import partial


class SACAgent(flax.struct.PyTreeNode):
    """SAC agent"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _aggregate_target_q(self, next_qs):
        """Reduce an (E, ...) ensemble of target Qs to (...).

        ``critic_target_aggregation``: ``mean_std`` (default) is the pessimistic
        ``mean - rho * std`` backup; ``min`` is the min over members (DICE-RL).
        """
        if self.config["critic_target_aggregation"] == "min":
            return next_qs.min(axis=0)
        return next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        # Shapes: B = batch size, T = chunk_size (sequence length), A = per-step
        # action dim, FA = full action dim the actor/critic operate on
        # (T*A if action_chunking else A), E = num_qs (critic ensemble size).
        # Batch comes from `sample_sequence` (flat obs) or
        # `sample_trivial_sequence` (T=1, flat or dict obs):
        #   observations: (B, ob_dim) or {pixels: (B, H, W, C, ncams),
        #     state: (B, state_dim)} [no time axis];
        #   next_observations: same but with a time axis at axis 1, i.e.
        #     (B, T, ob_dim) or a dict of (B, T, ...) leaves;
        #   actions (B, T, A); rewards/masks/terminals/valid (B, T).
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )  # (B, T, A) -> (B, T*A)
        else:
            batch_actions = batch["actions"][
                ..., 0, :
            ]  # take the first action: (B, T, A) -> (B, A)

        # Critic loss.
        rng, sample_rng = jax.random.split(rng, 2)
        # Select next_obs at the end of the chunk: (B, T, ob_dim) -> (B, ob_dim).
        next_observations = jax.tree_util.tree_map(
            lambda x: x[:, -1, ...], batch["next_observations"]
        )
        next_dist = self.network.select("actor")(next_observations)
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        next_qs = self.network.select("target_critic")(
            next_observations, next_actions
        )  # (E, B)
        next_q = self._aggregate_target_q(next_qs)  # (B,)
        if self.config["entropy_backup"]:
            next_q = next_q - self.network.select("alpha")() * next_log_probs

        # rewards[..., -1] is the discounted reward sum over the chunk: (B,)
        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** self.config["chunk_size"])
            * batch["masks"][..., -1]
            * next_q
        )  # (B,)
        target_q = jax.lax.stop_gradient(target_q)

        q = self.network.select("critic")(
            batch["observations"], batch_actions, params=grad_params
        )  # (E, B)
        # (E, B) - (B,) broadcasts over the ensemble; valid[..., -1] (B,) masks
        # chunks that crossed an episode boundary.
        critic_loss = (jnp.square(q - target_q) * batch["valid"][..., -1]).mean()

        total_loss = critic_loss

        return total_loss, {
            "total_loss": total_loss,
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "target_q": target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        # Only uses batch['observations'], which has no time axis in either
        # batch format: (B, ob_dim) flat, or a dict
        # {pixels: (B, H, W, C, ncams), state: (B, state_dim)}.

        # Actor loss.
        dist = self.network.select("actor")(
            batch["observations"], params=grad_params
        )  # dist over (B, FA)
        # Evaluate the density from the pre-tanh sample, including at saturation.
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        qs = self.network.select("critic")(batch["observations"], actions)  # (E, B)
        q = jnp.mean(qs, axis=0)  # (B,)

        actor_loss = (log_probs * self.network.select("alpha")() - q).mean()  # scalar

        # Entropy loss.
        alpha = self.network.select("alpha")(params=grad_params)  # scalar ()
        entropy = -jax.lax.stop_gradient(log_probs).mean()  # scalar
        alpha_loss = (
            alpha * (entropy - self.config["target_entropy"])
        ).mean()  # scalar

        total_loss = actor_loss + alpha_loss

        action_std = dist._distribution.stddev()  # pre-tanh base Normal: (B, FA)

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "entropy": -log_probs.mean(),
            "action_std": action_std.mean(),
            "q": q.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self,
        observations,
        rng=None,
        deterministic=False,
    ):
        # observations: (B, ob_dim) flat, or a dict
        # {pixels: (B, H, W, C, ncams), state: (B, state_dim)}.
        dist = self.network.select("actor")(observations)  # dist over (B, FA)
        if deterministic:
            actions = dist.mode()  # (B, FA), tanh of the base Normal's mean
        else:
            assert rng is not None, (
                "RNG key must be provided for stochastic action sampling."
            )
            actions = dist.sample(seed=rng)  # (B, FA)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=("use_target",))
    def compute_q_values(self, observations, actions, use_target=False):
        """Per-ensemble-member Q values, without the mean/std reduction used in the losses.

        Args:
            observations: (B, ob_dim) flat, or a dict
                {pixels: (B, H, W, C, ncams), state: (B, state_dim)}.
            actions: (B, FA), or (B, T, A) which is flattened to (B, T*A) --
                matching how `critic_loss` builds `batch_actions` under action
                chunking. No time axis is expected on the observations.
            use_target: Query 'target_critic' instead of 'critic'.

        Returns:
            (E, B) array of Q values, E = config['num_qs'].
        """
        if actions.ndim == 3:
            actions = jnp.reshape(
                actions, (actions.shape[0], -1)
            )  # (B, T, A) -> (B, T*A)
        module = "target_critic" if use_target else "critic"
        return self.network.select(module)(observations, actions)  # (E, B)

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_action_batch(self, observations, rng, num_samples):
        """Draw `num_samples` actions per observation from the actor's distribution.

        The actor is a tanh-squashed diagonal Gaussian, so the squashed
        distribution has no closed-form mean/std -- take empirical statistics over
        these samples instead. (The raw tfd distribution is available outside of
        jit via `agent.network.select('actor')(observations)`; its `_distribution`
        attribute is the pre-tanh Normal, whose `mean()`/`stddev()` are the
        parameters the actor head emits.)

        Args:
            observations: (B, ob_dim) flat, or a dict
                {pixels: (B, H, W, C, ncams), state: (B, state_dim)}.
            rng: PRNG key.
            num_samples: Static number of samples per observation.

        Returns:
            (num_samples, B, FA) actions, tanh-squashed and clipped to (-1, 1).
        """
        dist = self.network.select("actor")(observations)  # dist over (B, FA)
        actions = dist.sample(seed=rng, sample_shape=(num_samples,))  # (N, B, FA)
        return jnp.clip(actions, -1, 1)

    def save(self, path):
        """Save the full agent state (params, target params, optimizer state, rng) to `path`."""
        from nicerl.utils.atomic_io import atomic_pickle

        atomic_pickle(flax.serialization.to_state_dict(self), path)

    def load(self, path):
        """Return a new agent restored from `path`, using `self` as the target structure.

        The agent must already be built (via `create` with matching shapes/config).
        """
        from nicerl.utils.atomic_io import load_pickle

        state = load_pickle(path)
        return flax.serialization.from_state_dict(self, state)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["chunk_size"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config["target_entropy"] is None:
            config["target_entropy"] = (
                -config["target_entropy_multiplier"] * full_action_dim
            )

        print(
            f"[sac] action_dim: {action_dim}, full_action_dim: {full_action_dim}, target_entropy: {config['target_entropy']}"
        )

        encoder_def = None
        if config["use_encoder"]:
            encoder_def = PixelStateEncoder(
                features=config["cnn_features"],
                strides=config["cnn_strides"],
                padding=config["cnn_padding"],
                use_bottleneck=config["use_bottleneck"],
                bottleneck_dim=config["bottleneck_dim"],
            )
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
        )
        critic_def = ObservationActionMultiplexer(
            network=critic_def,
            obs_encoder=copy.deepcopy(encoder_def),
            action_encoder=None,
        )
        actor_base_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activate_final=True,
            layer_norm=config["actor_layer_norm"],
        )
        actor_def = TanhNormal(actor_base_cls, full_action_dim)
        actor_def = ObservationActionMultiplexer(
            network=actor_def,
            obs_encoder=copy.deepcopy(encoder_def),
            action_encoder=None,
            do_combine_encoding=True,
        )
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="sac",  # Agent name.
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Action dimension (will be set automatically).
            ## Common hyperparamters
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=True,  # LayerNorm after every hidden activation of the actor MLP.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,
            ## Q-chunking hyperparameters
            chunk_size=ml_collections.config_dict.placeholder(int),  # Will be set
            action_chunking=False,  # Use Q-chunking or just n-step return
            ## RL hyperparameters
            num_qs=10,  # Critic ensemble size
            rho=0.5,  # Pessimistic backup
            critic_target_aggregation="mean_std",  # 'mean_std' (mean - rho * std) or 'min' over the ensemble
            entropy_backup=False,  # Enable for standard DSRL-SAC; preserve historical runs.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ## Other design variants/hyperparameters
            target_entropy=ml_collections.config_dict.placeholder(
                float
            ),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            # Image Encoder
            use_encoder=False,  # Whether to use an image encoder.
            cnn_features=(32, 32, 32, 32),  # CNN features for image encoder.
            cnn_strides=(2, 2, 2, 2),  # CNN strides for image encoder.
            cnn_padding="VALID",  # CNN padding for image encoder.
            use_bottleneck=True,  # Whether to use a bottleneck encoded features.
            bottleneck_dim=50,  # Latent feature dimension post-bottleneck.
        )
    )
    return config
