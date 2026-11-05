"""Critic-only best-of-N over cached, frozen VLA action chunks."""

import copy
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from nicerl.agents.sac import SACAgent
from nicerl.utils.encoders.networks import (
    ObservationActionMultiplexer,
    PixelStateEncoder,
)
from nicerl.utils.flax_utils import ModuleDict, TrainState
from nicerl.utils.networks import Value


class BFNAgent(SACAgent):
    """Reuse SAC's update/persistence machinery; construct only Q and target Q.

    The policy is argmax over the online ensemble mean (optionally minus
    selection_rho * std). The SAME policy selects next actions for the backup;
    the target ensemble evaluates those actions with the existing rho penalty.
    There is no actor, entropy term, or behavior-cloning loss.
    """

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(self, observations, base_actions, rng=None, deterministic=False):
        del rng, deterministic  # Only the supplied base samples are stochastic.
        batch_size, n, action_dim = base_actions.shape
        if (
            n != self.config["n_base_samples"]
            or action_dim != self.config["action_dim"]
        ):
            raise ValueError(
                "BFN candidates must have shape (batch, n_base_samples, action_dim)"
            )

        # Vmap only actions: the shared observation encoder is not expanded to
        # a B*N pixel batch. Values have shape (num_qs, batch, N).
        qs = jax.vmap(
            lambda actions: self.network.select("critic")(
                observations, actions
            ).reshape(self.config["num_qs"], batch_size),
            in_axes=1,
            out_axes=-1,
        )(base_actions)
        scores = qs.mean(axis=0) - self.config["selection_rho"] * qs.std(axis=0)
        best = jnp.argmax(scores, axis=-1)
        return base_actions[jnp.arange(batch_size), best]

    def critic_loss(self, batch, grad_params, rng):
        cached = batch["next_base_actions"]
        n = self.config["n_base_samples"]
        if cached.shape[1] < n:
            raise ValueError("Replay must contain at least n_base_samples next actions")
        if cached.shape[1] == n:
            candidates = cached
        else:
            # Support larger historical caches without changing the policy's N.
            indices = jnp.argsort(jax.random.uniform(rng, cached.shape[:2]), axis=1)[
                :, :n
            ]
            candidates = jnp.take_along_axis(cached, indices[..., None], axis=1)
        next_actions = self.sample_actions(batch["next_observations"], candidates)
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], next_actions
        ).reshape(self.config["num_qs"], -1)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)
        target = jax.lax.stop_gradient(
            batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        )
        qs = self.network.select("critic")(
            batch["observations"], batch["actions"], params=grad_params
        ).reshape(self.config["num_qs"], -1)
        loss = jnp.square(qs - target).mean()
        return loss, {
            "critic/total_loss": loss,
            "critic/critic_loss": loss,
            "critic/q_mean": qs.mean(),
            "critic/q_min": qs.min(),
            "critic/q_max": qs.max(),
            "critic/target_q": target.mean(),
            "bfn/n_base_samples": jnp.asarray(n),
            "bfn/selected_next_q": next_q.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None):
        return self.critic_loss(batch, grad_params, self.rng if rng is None else rng)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = copy.deepcopy(config)
        action_dim = ex_actions.shape[-1]
        if config["n_base_samples"] < 1 or config["num_qs"] < 1:
            raise ValueError("n_base_samples and num_qs must be positive")
        if action_dim != config["action_horizon"] * config["raw_action_dim"]:
            raise ValueError("BFN expects one flattened action chunk per transition")
        config["action_dim"] = action_dim
        encoder = None
        if config["use_encoder"]:
            encoder = PixelStateEncoder(
                features=config["cnn_features"],
                strides=config["cnn_strides"],
                padding=config["cnn_padding"],
                use_bottleneck=config["use_bottleneck"],
                bottleneck_dim=config["bottleneck_dim"],
            )
        critic = ObservationActionMultiplexer(
            network=Value(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["value_layer_norm"],
                num_ensembles=config["num_qs"],
            ),
            obs_encoder=encoder,
            action_encoder=None,
        )
        model = ModuleDict({"critic": critic, "target_critic": copy.deepcopy(critic)})
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        args = (ex_observations, ex_actions)
        params = model.init(init_rng, critic=args, target_critic=args)["params"]
        params["modules_target_critic"] = params["modules_critic"]
        network = TrainState.create(model, params, tx=optax.adam(config["lr"]))
        return cls(rng, network=network, config=flax.core.FrozenDict(config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="bfn",
            n_base_samples=8,
            selection_rho=0.0,
            lr=3e-4,
            batch_size=256,
            discount=0.975,
            tau=0.005,
            num_qs=10,
            rho=0.5,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            use_encoder=True,
            cnn_features=(32, 32, 32, 32),
            cnn_strides=(2, 2, 2, 2),
            cnn_padding="VALID",
            use_bottleneck=True,
            bottleneck_dim=50,
            action_horizon=ml_collections.config_dict.placeholder(int),
            raw_action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
