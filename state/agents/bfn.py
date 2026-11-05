import copy
from functools import partial
from typing import Any
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value


class BFNAgent(flax.struct.PyTreeNode):
    """Actor-critic with a best-of-N frozen base-policy actor."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    base_policy: Any

    def _normalize_batch(self, batch):
        """Convert sequence/action-chunk batches into flat-chunk transitions."""
        if batch["actions"].ndim == batch["observations"].ndim + 1:
            batch = dict(batch)
            b = batch["actions"].shape[0]
            batch["actions"] = batch["actions"].reshape(b, -1)
            batch["rewards"] = batch["rewards"][..., -1]
            if batch["masks"].ndim > 1:
                batch["masks"] = batch["masks"][..., -1]
            if "terminals" in batch and batch["terminals"].ndim > 1:
                batch["terminals"] = batch["terminals"][..., -1]
            if batch["next_observations"].ndim == batch["observations"].ndim + 1:
                batch["next_observations"] = batch["next_observations"][..., -1, :]
        return batch

    def _discount(self, batch=None):
        """Per-row ``discount ** n_steps`` when the batch carries ``n_steps``, per-row
        ``discounts`` when it carries those, else the fixed chunk discount (see
        ``SACAgent._discount``)."""
        if batch is not None and "n_steps" in batch:
            return self.config["discount"] ** batch["n_steps"]
        if batch is not None and "discounts" in batch:
            return batch["discounts"]
        return self.config["discount"] ** self.config["horizon_length"]

    def _clip_actions(self, actions):
        bounds = (
            jnp.full((self.config["action_dim"],), self.config["action_clip"])
            .at[-1]
            .set(self.config["gripper_action_clip"])
        )
        bounds = jnp.tile(bounds, self.config["horizon_length"])
        return jnp.clip(actions, -bounds, bounds)

    def _aggregate_q(self, qs):
        return qs.mean(axis=0) - self.config["critic_target_rho"] * qs.std(axis=0)
        return qs.mean(axis=0)

    def _sample_base_actions_n(self, observations, rng):
        """Sample base-policy chunks as an (N, B, H*A) tensor."""
        n = self.config["n_base_samples"]
        if n == 1:
            base_actions = self.base_policy.sample_actions(observations, seed=rng)
            if base_actions.ndim < observations.ndim:
                base_actions = base_actions[None]
            base_actions_n = self._clip_actions(base_actions)[None]
        else:
            batch_size = observations.shape[0]
            obs_tiled = jnp.repeat(observations[None, ...], n, axis=0)
            obs_tiled = obs_tiled.reshape((n * batch_size,) + observations.shape[1:])
            base_actions = self.base_policy.sample_actions(obs_tiled, seed=rng)
            base_actions = self._clip_actions(base_actions)
            base_actions_n = base_actions.reshape(n, batch_size, base_actions.shape[-1])
        return jax.lax.stop_gradient(base_actions_n)

    def _select_best_actions(self, observations, base_actions_n, critic_name):
        """Select the base-policy sample with the highest configured ensemble value."""
        n = self.config["n_base_samples"]
        batch_size = observations.shape[0]
        action_dim = base_actions_n.shape[-1]
        actions = base_actions_n.transpose(1, 0, 2)
        obs_tiled = jnp.repeat(observations[:, None, :], n, axis=1)
        obs_tiled = obs_tiled.reshape((batch_size * n,) + observations.shape[1:])
        action_tiled = actions.reshape(batch_size * n, action_dim)
        qs = self.network.select(critic_name)(obs_tiled, action_tiled)
        if qs.ndim == 1:
            qs = qs[None]
        q = self._aggregate_q(qs).reshape(batch_size, n)
        best_idxs = jnp.argmax(q, axis=1)
        best_actions = actions[jnp.arange(batch_size), best_idxs]
        return (best_actions, q)

    def critic_loss(self, batch, grad_params, rng):
        """Compute TD critic loss using best-of-N base-policy target actions."""
        batch_size = batch["rewards"].shape[0]
        rng, next_rng, subsample_rng = jax.random.split(rng, 3)
        next_base_actions_n = self._sample_base_actions_n(
            batch["next_observations"], next_rng
        )
        next_actions, next_candidate_q = self._select_best_actions(
            batch["next_observations"], next_base_actions_n, "critic"
        )
        target_next_qs = self.network.select("target_critic")(
            batch["next_observations"], next_actions
        )
        if target_next_qs.ndim == 1:
            target_next_qs = target_next_qs[None]
        if self.config["critic_subsample_size"] is not None:
            subsample_idxs = jax.random.choice(
                subsample_rng,
                self.config["critic_ensemble_size"],
                shape=(self.config["critic_subsample_size"],),
                replace=False,
            )
            target_next_qs = target_next_qs[subsample_idxs]
        target_next_q = self._aggregate_q(target_next_qs)
        target_q = (
            batch["rewards"] + self._discount(batch) * batch["masks"] * target_next_q
        )
        if (
            self.config["critic_target_min"] is not None
            or self.config["critic_target_max"] is not None
        ):
            target_q = jnp.clip(
                target_q,
                self.config["critic_target_min"],
                self.config["critic_target_max"],
            )
        predicted_qs = self.network.select("critic")(
            batch["observations"], batch["actions"], params=grad_params
        )
        if predicted_qs.ndim == 1:
            predicted_qs = predicted_qs[None]
        target_qs = jnp.broadcast_to(
            target_q, (self.config["critic_ensemble_size"], batch_size)
        )
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
        return (
            critic_loss,
            {
                "critic_loss": critic_loss,
                "predicted_qs": predicted_qs.mean(),
                "target_qs": target_qs.mean(),
                "target_candidate_q": next_candidate_q.mean(),
                "rewards": batch["rewards"].mean(),
            },
        )

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: (
                p * self.config["soft_target_update_rate"]
                + tp * (1 - self.config["soft_target_update_rate"])
            ),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def total_loss(
        self,
        batch,
        grad_params,
        rng=None,
        update_critic=True,
        update_actor=True,
    ):
        """Compute the critic loss."""
        del update_actor
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng = jax.random.split(rng)
        batch = self._normalize_batch(batch)
        critic_loss = 0.0
        if update_critic:
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
            for k, v in critic_info.items():
                info[f"critic/{k}"] = v
        return (critic_loss, info)

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def update(self, batch, update_critic=True, update_actor=True):
        """Update the critic and target critic; the base policy stays frozen."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(
                batch,
                grad_params,
                rng=rng,
                update_critic=update_critic,
                update_actor=update_actor,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if update_critic:
            self.target_update(new_network, "critic")
        return (self.replace(network=new_network, rng=new_rng), info)

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
        """Sample N base-policy chunks and return the one with highest Q."""
        del temperature, argmax
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        base_actions_n = self._sample_base_actions_n(observations, seed)
        actions, _ = self._select_best_actions(observations, base_actions_n, "critic")
        actions = self._clip_actions(actions)
        return actions[0] if unbatched else actions

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions_with_diagnostics(
        self, observations, seed=None, temperature=1.0, argmax=False
    ):
        """Expose the selected action's mean Q for standard Q-versus-return videos."""
        observations = jnp.asarray(observations)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        actions = self.sample_actions(observations, seed, temperature, argmax)
        qs = self.network.select("critic")(observations, actions)
        q = qs.mean(axis=0)
        if unbatched:
            return (actions[0], {"q": q[0]})
        return (actions, {"q": q})

    @jax.jit
    def sample_base_actions_only(self, observations, seed=None):
        """Sample one chunk from the base policy without Q selection."""
        return self.base_policy.sample_actions(observations, seed=seed)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, base_policy):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        config = copy.deepcopy(config)
        assert config["n_base_samples"] >= 1
        raw_action_dim = ex_actions.shape[-1]
        horizon_length = config["horizon_length"]
        chunk_action_dim = horizon_length * raw_action_dim
        ex_chunk_actions = jnp.zeros((ex_observations.shape[0], chunk_action_dim))
        config["horizon_length"] = horizon_length
        config["raw_action_dim"] = raw_action_dim
        config["chunk_action_dim"] = chunk_action_dim
        config["action_dim"] = raw_action_dim
        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=config["critic_ensemble_size"],
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_chunk_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_chunk_actions),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            base_policy=base_policy,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="bfn",
            lr=0.0003,
            batch_size=256,
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512), layer_norm=True
            ),
            discount=0.99,
            soft_target_update_rate=0.005,
            critic_ensemble_size=2,
            critic_subsample_size=None,
            critic_target_rho=0.5,
            critic_target_min=ml_collections.config_dict.placeholder(float),
            critic_target_max=ml_collections.config_dict.placeholder(float),
            action_clip=1.0,
            gripper_action_clip=1.0,
            dataset_action_clip_eps=1e-05,
            n_base_samples=1,
            horizon_length=ml_collections.config_dict.placeholder(int),
            raw_action_dim=ml_collections.config_dict.placeholder(int),
            chunk_action_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
