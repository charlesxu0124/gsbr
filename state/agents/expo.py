import copy
from functools import partial
from typing import Any
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, GeqLagrangeMultiplier, Value


class EXPOAgent(flax.struct.PyTreeNode):
    """EXPO with a frozen base policy and a learned residual edit policy."""

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

    def _discount(self, batch):
        """Per-row ``discount ** n_steps`` when the batch carries ``n_steps``, else
        the fixed chunk discount (see ``SACAgent._discount``)."""
        if "n_steps" in batch:
            return self.config["discount"] ** batch["n_steps"]
        return self.config["discount"] ** self.config["horizon_length"]

    def _temperature(self, grad_params=None):
        return self.network.select("temperature")(params=grad_params)

    def _critic(self, observations, actions, params=None, critic_name="critic"):
        qs = self.network.select(critic_name)(observations, actions, params=params)
        if qs.ndim == 1:
            qs = qs[None, :]
        return qs

    def _action_bounds(self):
        raw_bounds = (
            jnp.full((self.config["action_dim"],), self.config["action_clip"])
            .at[-1]
            .set(self.config["gripper_action_clip"])
        )
        return jnp.tile(raw_bounds, self.config["horizon_length"])

    def _clip_actions(self, actions):
        bounds = self._action_bounds()
        return jnp.clip(actions, -bounds, bounds)

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

    def _action_diversity(self, actions_n):
        per_state = jnp.sqrt(jnp.mean(jnp.var(actions_n, axis=0), axis=-1) + 1e-08)
        return per_state

    def _nearest_ref_deviation(self, actions, base_actions_n):
        per_candidate_mse = jnp.square(actions[None] - base_actions_n).mean(axis=-1)
        nearest_mse = per_candidate_mse.min(axis=0)
        return jnp.sqrt(nearest_mse + 1e-08)

    def _residual_dist(self, observations, base_actions, params=None):
        inputs = jnp.concatenate([observations, base_actions], axis=-1)
        return self.network.select("actor")(
            inputs, temperature=self.config["policy_temperature"], params=params
        )

    def _scale_log_probs(self, log_probs, action_dim):
        return log_probs - action_dim * jnp.log(self.config["residual_action_scale"])

    def _sample_residual_edits(
        self, observations, base_actions_m, rng, params=None, argmax=False
    ):
        """Edit an (M, B, A) tensor of base actions with the residual actor."""
        m = self.config["n_residual_samples"]
        batch_size = observations.shape[0]
        action_dim = base_actions_m.shape[-1]
        obs_flat = jnp.repeat(observations[None, ...], m, axis=0)
        obs_flat = obs_flat.reshape((m * batch_size,) + observations.shape[1:])
        base_flat = base_actions_m.reshape(m * batch_size, action_dim)
        dist = self._residual_dist(obs_flat, base_flat, params=params)
        residual = dist.mode() if argmax else dist.sample(seed=rng)
        log_probs = dist.log_prob(residual)
        log_probs = self._scale_log_probs(log_probs, action_dim)
        scaled_residual = residual * self.config["residual_action_scale"]
        edited = self._clip_actions(base_flat + scaled_residual)
        return (
            edited.reshape(m, batch_size, action_dim),
            residual.reshape(m, batch_size, action_dim),
            log_probs.reshape(m, batch_size),
        )

    def _sample_candidates(self, observations, rng, argmax_residual=False):
        """Sample N base actions and M residual-edited candidates."""
        base_rng, residual_rng = jax.random.split(rng)
        base_actions_n = self._sample_base_actions_n(observations, base_rng)
        candidates = base_actions_n.transpose(1, 0, 2)
        if self.config["n_residual_samples"] > 0:
            base_actions_m = base_actions_n[: self.config["n_residual_samples"]]
            edited_m, _, _ = self._sample_residual_edits(
                observations, base_actions_m, residual_rng, argmax=argmax_residual
            )
            candidates = jnp.concatenate(
                [candidates, edited_m.transpose(1, 0, 2)], axis=1
            )
        return candidates

    def _select_best_actions(self, observations, candidates, critic_name):
        """Select the candidate with the highest ensemble min-Q."""
        batch_size, num_candidates, action_dim = candidates.shape
        obs_tiled = jnp.repeat(observations[:, None, :], num_candidates, axis=1)
        obs_tiled = obs_tiled.reshape(
            (batch_size * num_candidates,) + observations.shape[1:]
        )
        action_tiled = candidates.reshape(batch_size * num_candidates, action_dim)
        qs = self._critic(obs_tiled, action_tiled, critic_name=critic_name)
        q = qs.min(axis=0).reshape(batch_size, num_candidates)
        best_idxs = jnp.argmax(q, axis=1)
        best_actions = candidates[jnp.arange(batch_size), best_idxs]
        return (best_actions, q)

    def _select_best_actions_with_indices(self, observations, candidates, critic_name):
        """Select the candidate with the highest ensemble min-Q and return indices."""
        batch_size, num_candidates, action_dim = candidates.shape
        obs_tiled = jnp.repeat(observations[:, None, :], num_candidates, axis=1)
        obs_tiled = obs_tiled.reshape(
            (batch_size * num_candidates,) + observations.shape[1:]
        )
        action_tiled = candidates.reshape(batch_size * num_candidates, action_dim)
        qs = self._critic(obs_tiled, action_tiled, critic_name=critic_name)
        q = qs.min(axis=0).reshape(batch_size, num_candidates)
        best_idxs = jnp.argmax(q, axis=1)
        best_actions = candidates[jnp.arange(batch_size), best_idxs]
        return (best_actions, q, best_idxs)

    def _sample_candidates_with_refs(self, observations, rng, argmax_residual=False):
        base_rng, residual_rng = jax.random.split(rng)
        base_actions_n = self._sample_base_actions_n(observations, base_rng)
        candidates = base_actions_n.transpose(1, 0, 2)
        refs = candidates
        if self.config["n_residual_samples"] > 0:
            base_actions_m = base_actions_n[: self.config["n_residual_samples"]]
            edited_m, _, _ = self._sample_residual_edits(
                observations, base_actions_m, residual_rng, argmax=argmax_residual
            )
            edited_candidates = edited_m.transpose(1, 0, 2)
            candidates = jnp.concatenate([candidates, edited_candidates], axis=1)
            refs = jnp.concatenate([refs, base_actions_m.transpose(1, 0, 2)], axis=1)
        return (candidates, refs, base_actions_n)

    def critic_loss(self, batch, grad_params, rng):
        """Compute TD critic loss using EXPO best-of-(N+M) target actions."""
        batch_size = batch["rewards"].shape[0]
        rng, candidate_rng, subsample_rng = jax.random.split(rng, 3)
        next_candidates = self._sample_candidates(
            batch["next_observations"], candidate_rng
        )
        next_actions, next_candidate_q = self._select_best_actions(
            batch["next_observations"], next_candidates, "target_critic"
        )
        target_next_qs = self._critic(
            batch["next_observations"], next_actions, critic_name="target_critic"
        )
        if self.config["critic_subsample_size"] is not None:
            subsample_idxs = jax.random.choice(
                subsample_rng,
                self.config["critic_ensemble_size"],
                shape=(self.config["critic_subsample_size"],),
                replace=False,
            )
            target_next_qs = target_next_qs[subsample_idxs]
        target_next_q = target_next_qs.mean(axis=0) - self.config[
            "critic_target_rho"
        ] * target_next_qs.std(axis=0)
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
        predicted_qs = self._critic(
            batch["observations"], batch["actions"], params=grad_params
        )
        target_qs = jnp.broadcast_to(
            target_q, (self.config["critic_ensemble_size"], batch_size)
        )
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
        return (
            critic_loss,
            {
                "critic/critic_loss": critic_loss,
                "critic/predicted_qs": predicted_qs.mean(),
                "critic/target_qs": target_qs.mean(),
                "critic/target_candidate_q": next_candidate_q.mean(),
                "critic/rewards": batch["rewards"].mean(),
            },
        )

    def actor_loss(self, batch, grad_params, rng):
        """Train the residual actor around a batch or fresh base-policy action."""
        reference_rng, actor_rng = jax.random.split(rng)
        reference_actions = self.base_policy.sample_actions(
            batch["observations"], seed=reference_rng
        )
        if reference_actions.ndim < batch["observations"].ndim:
            reference_actions = reference_actions[None]
        reference_actions = jax.lax.stop_gradient(self._clip_actions(reference_actions))
        dist = self._residual_dist(
            batch["observations"], reference_actions, params=grad_params
        )
        residual, log_probs = dist.sample_and_log_prob(seed=actor_rng)
        log_probs = self._scale_log_probs(log_probs, residual.shape[-1])
        scaled_residual = residual * self.config["residual_action_scale"]
        edited_actions = self._clip_actions(reference_actions + scaled_residual)
        predicted_qs = self._critic(batch["observations"], edited_actions)
        predicted_q = predicted_qs.mean(axis=0)
        temperature = self._temperature()
        actor_loss = -predicted_q.mean()
        action_std = dist._distribution.stddev().mean()
        ref_deviation = jnp.sqrt(
            jnp.mean(jnp.square(edited_actions - batch["actions"]), axis=-1) + 1e-08
        )
        ref_diversity = jnp.zeros_like(ref_deviation)
        return (
            actor_loss,
            {
                "actor/total_loss": actor_loss,
                "actor/actor_loss": actor_loss,
                "actor/temperature": temperature,
                "actor/entropy": -log_probs.mean(),
                "actor/std": action_std,
                "actor/q": predicted_q.mean(),
                "actor/residual_action_scale": jnp.asarray(
                    self.config["residual_action_scale"]
                ),
                "actor/residual_magnitude": jnp.abs(scaled_residual).mean(),
                "actor/reference_action_magnitude": jnp.abs(reference_actions).mean(),
                "actor/edited_action_magnitude": jnp.abs(edited_actions).mean(),
                "actor/n_base_samples": jnp.asarray(self.config["n_base_samples"]),
                "actor/n_residual_samples": jnp.asarray(
                    self.config["n_residual_samples"]
                ),
                "actor/ref_action_diversity": ref_diversity.mean(),
                "actor/ref_action_diversity_max": ref_diversity.max(),
                "actor/rl_ref_deviation": ref_deviation.mean(),
                "actor/rl_ref_deviation_max": ref_deviation.max(),
            },
        )

    def temperature_loss(self, batch, grad_params, rng):
        reference_rng, actor_rng = jax.random.split(rng)
        reference_actions = self.base_policy.sample_actions(
            batch["observations"], seed=reference_rng
        )
        if reference_actions.ndim < batch["observations"].ndim:
            reference_actions = reference_actions[None]
        reference_actions = jax.lax.stop_gradient(self._clip_actions(reference_actions))
        dist = self._residual_dist(batch["observations"], reference_actions)
        residual, log_probs = dist.sample_and_log_prob(seed=actor_rng)
        log_probs = self._scale_log_probs(log_probs, residual.shape[-1])
        entropy = -log_probs.mean()
        temp_loss = self.network.select("temperature")(
            lhs=entropy, rhs=self.config["target_entropy"], params=grad_params
        )
        return (
            temp_loss,
            {
                "temperature/temperature_loss": temp_loss,
                "temperature/temperature": self._temperature(),
                "temperature/entropy": entropy,
            },
        )

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss, batch),
            "actor": partial(self.actor_loss, batch),
            "temperature": partial(self.temperature_loss, batch),
        }

    def target_update(self, network, module_name):
        """Polyak update a target module."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: (
                p * self.config["soft_target_update_rate"]
                + tp * (1 - self.config["soft_target_update_rate"])
            ),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
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
        """Compute losses for validation/evaluation compatibility."""

        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng, temp_rng = jax.random.split(rng, 4)
        batch = self._normalize_batch(batch)
        critic_loss = 0.0
        if update_critic:
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
            info.update(critic_info)
        actor_loss = 0.0
        temp_loss = 0.0
        if update_actor:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            info.update(actor_info)
        return (critic_loss + actor_loss + temp_loss, info)

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def update(self, batch, update_critic=True, update_actor=True):
        """Update the critic and residual actor; the base policy stays frozen."""

        new_rng, rng = jax.random.split(self.rng)
        batch = self._normalize_batch(batch)
        loss_fns = self.loss_fns(batch)
        if not update_critic:
            loss_fns.pop("critic")
        if not update_actor:
            loss_fns.pop("actor")
            loss_fns.pop("temperature")
        else:
            loss_fns["temperature"] = lambda params, rng: (jnp.array(0.0), {})
        new_network, info_tree = self.network.apply_loss_fns(
            loss_fns, rng=rng, has_aux=True
        )
        info = {}
        for loss_info in info_tree.values():
            info.update(loss_info)
        if update_critic:
            self.target_update(new_network, "critic")
        return (self.replace(network=new_network, rng=new_rng), info)

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
        """Sample EXPO OTF actions by ranking base and residual candidates."""
        del temperature
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        candidates = self._sample_candidates(observations, seed, argmax_residual=argmax)
        actions, _ = self._select_best_actions(
            observations, candidates, "target_critic"
        )
        actions = self._clip_actions(actions)
        return actions[0] if unbatched else actions

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions_with_diagnostics(
        self, observations, seed=None, temperature=1.0, argmax=False
    ):
        """Sample EXPO OTF actions and diagnostics from the same candidates."""
        del temperature
        observations = jnp.asarray(observations)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        candidates, refs, base_actions_n = self._sample_candidates_with_refs(
            observations, seed, argmax_residual=argmax
        )
        actions, _, best_idxs = self._select_best_actions_with_indices(
            observations, candidates, "target_critic"
        )
        actions = self._clip_actions(actions)
        selected_refs = refs[jnp.arange(observations.shape[0]), best_idxs]
        diagnostics = {
            "q": self._critic(observations, actions, critic_name="critic").mean(axis=0),
            "ref_action_diversity": self._action_diversity(base_actions_n),
            "rl_ref_deviation": jnp.sqrt(
                jnp.mean(jnp.square(actions - selected_refs), axis=-1) + 1e-08
            ),
        }
        if unbatched:
            actions = actions[0]
            diagnostics = jax.tree_util.tree_map(lambda x: x[0], diagnostics)
        return (actions, diagnostics)

    @jax.jit
    def sample_base_actions_only(self, observations, seed=None):
        """Sample one chunk from the base policy without residual edits."""
        return self.base_policy.sample_actions(observations, seed=seed)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, base_policy):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        config = copy.deepcopy(config)
        assert config["critic_target_rho"] >= 0
        if (
            config["critic_target_min"] is not None
            and config["critic_target_max"] is not None
        ):
            assert config["critic_target_min"] < config["critic_target_max"]
        raw_action_dim = ex_actions.shape[-1]
        horizon_length = config["horizon_length"]
        chunk_action_dim = horizon_length * raw_action_dim
        ex_chunk_actions = jnp.zeros((ex_observations.shape[0], chunk_action_dim))
        ex_actor_inputs = jnp.concatenate([ex_observations, ex_chunk_actions], axis=-1)
        config["horizon_length"] = horizon_length
        config["raw_action_dim"] = raw_action_dim
        config["chunk_action_dim"] = chunk_action_dim
        config["action_dim"] = raw_action_dim
        if config["target_entropy"] is None:
            config["target_entropy"] = (
                -config["target_entropy_multiplier"] * chunk_action_dim
            )
        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=config["critic_ensemble_size"],
        )
        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=chunk_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config["actor_fc_scale"],
        )
        temperature_def = GeqLagrangeMultiplier(
            init_value=config["temperature_init"], constraint_shape=()
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_chunk_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_chunk_actions),
            ),
            actor=(actor_def, (ex_actor_inputs,)),
            temperature=(temperature_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = {
            "critic": optax.adam(learning_rate=config["lr"]),
            "actor": optax.adam(learning_rate=config["lr"]),
            "temperature": optax.adam(learning_rate=config["lr"]),
        }
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
            agent_name="expo",
            actor_update_period=1,
            lr=0.0003,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512), layer_norm=True
            ),
            actor_layer_norm=True,
            discount=0.99,
            soft_target_update_rate=0.005,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            temperature_init=1.0,
            actor_fc_scale=1.0,
            critic_ensemble_size=2,
            critic_subsample_size=None,
            critic_target_rho=0.5,
            critic_target_min=ml_collections.config_dict.placeholder(float),
            critic_target_max=ml_collections.config_dict.placeholder(float),
            policy_temperature=1.0,
            n_base_samples=8,
            n_residual_samples=8,
            residual_action_scale=0.1,
            dataset_action_clip_eps=1e-05,
            action_clip=1.0,
            gripper_action_clip=1.0,
            horizon_length=ml_collections.config_dict.placeholder(int),
            raw_action_dim=ml_collections.config_dict.placeholder(int),
            chunk_action_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
