from dataclasses import field
from functools import partial
from typing import Any, Dict, Optional, Sequence
import chex
import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


def flatten_ref_actions(ref_actions, batch_size):
    """Normalize reference actions to (B, N*H*A).

    ``ref_actions`` may be passed either already flat as ``(B, N*H*A)`` or as
    sampled chunks ``(N, B, H*A)``.  The sampled form intentionally uses the
    leading axis for the number of reference samples.
    """
    if ref_actions.ndim == 3:
        assert ref_actions.shape[1] == batch_size, (
            "Expected ref_actions shape (N, B, H*A), got a different batch axis."
        )
        batch_size = ref_actions.shape[1]
        return ref_actions.transpose(1, 0, 2).reshape(batch_size, -1)
    assert ref_actions.shape[0] == batch_size, (
        "Expected flat ref_actions shape (B, N*H*A)."
    )
    return ref_actions


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activation: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activation: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) and self.layer_norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class ResidualMLPBlock(nn.Module):
    """Two-layer residual block for fixed-width vector features."""

    hidden_dim: int
    activation: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(x)
        x = self.activation(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(x)
        x = residual + x
        x = self.activation(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        return x


class ResidualMLP(nn.Module):
    """OCBench depth convention: two hidden entries per residual block.

    Input/output projections are not counted. For example, ``(1024,) * 8``
    has four two-layer blocks, matching ``(1024,) * 4`` in ogbench_plus.
    Keep the block module names stable so legacy GSBR weights still load.
    """

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        if len(self.hidden_dims) % 2:
            raise ValueError("ResidualMLP requires an even number of hidden layers")
        if not self.hidden_dims:
            return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)
        if any(dim != self.hidden_dims[0] for dim in self.hidden_dims):
            raise ValueError("ResidualMLP requires constant hidden widths")
        x = nn.Dense(self.hidden_dims[0], kernel_init=self.kernel_init)(x)
        x = self.activation(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        for hidden_dim in self.hidden_dims[::2]:
            x = ResidualMLPBlock(
                hidden_dim=hidden_dim,
                activation=self.activation,
                kernel_init=self.kernel_init,
                layer_norm=self.layer_norm,
            )(x)
        return nn.Dense(self.output_dim, kernel_init=self.kernel_init)(x)


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class LagrangeMultiplier(nn.Module):
    """Lagrange multiplier module (copied from hil-serl)."""

    init_value: float = 1.0
    constraint_shape: Sequence[int] = ()
    constraint_type: str = "eq"
    parameterization: Optional[str] = None

    @nn.compact
    def __call__(
        self, *, lhs: Optional[jnp.ndarray] = None, rhs: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        init_value = self.init_value
        if self.constraint_type != "eq":
            assert init_value > 0, (
                "Inequality constraints must have non-negative initial multiplier values"
            )
            if self.parameterization == "softplus":
                init_value = jnp.log(jnp.exp(init_value) - 1)
            elif self.parameterization == "exp":
                init_value = jnp.log(init_value)
            elif self.parameterization == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid multiplier parameterization {self.parameterization}"
                )
        else:
            assert self.parameterization is None, (
                "Equality constraints must have no parameterization"
            )
        multiplier = self.param(
            "lagrange",
            lambda _, shape: jnp.full(shape, init_value),
            self.constraint_shape,
        )
        if self.constraint_type != "eq":
            if self.parameterization == "softplus":
                multiplier = nn.softplus(multiplier)
            elif self.parameterization == "exp":
                multiplier = jnp.exp(multiplier)
            elif self.parameterization == "none":
                pass
            else:
                raise ValueError(
                    f"Invalid multiplier parameterization {self.parameterization}"
                )
        if lhs is None:
            return multiplier
        if rhs is None:
            rhs = jnp.zeros_like(lhs)
        diff = lhs - rhs
        chex.assert_equal_shape([diff, multiplier])
        if self.constraint_type == "eq":
            return multiplier * diff
        elif self.constraint_type == "geq":
            return multiplier * diff
        elif self.constraint_type == "leq":
            return -multiplier * diff


GeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="geq", parameterization="softplus"
)
LeqLagrangeMultiplier = partial(
    LagrangeMultiplier, constraint_type="leq", parameterization="softplus"
)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        activation: Activation function (default: nn.gelu).
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    activation: Any = nn.gelu
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 0.01
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
            activation=self.activation,
        )
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        elif not self.const_std:
            self.log_stds = self.param(
                "log_stds", nn.initializers.zeros, (self.action_dim,)
            )

    def __call__(
        self,
        observations,
        temperature=1.0,
        ref_actions=None,
        ref_action_dropout_rng=None,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
            ref_actions: Reference actions to condition on (optional). Shape should
                be (N, B, H*A) for sampled chunks, or (B, N*H*A) if already flat.
            ref_action_dropout_rng: RNG for dropping the full reference action input.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        if ref_actions is not None:
            ref_actions = flatten_ref_actions(ref_actions, inputs.shape[0])
            inputs = jnp.concatenate([inputs, ref_actions], axis=-1)
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        elif self.const_std:
            log_stds = jnp.zeros_like(means)
        else:
            log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )
        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    num_ensembles: int = 2
    encoder: nn.Module = None
    network_class: str = "MLP"
    network_kwargs: Dict[str, Any] = field(default_factory=dict)

    def setup(self):
        network = MLP
        network_args = {
            "hidden_dims": (*self.network_kwargs["hidden_dims"], 1),
            "activate_final": False,
            "layer_norm": self.network_kwargs["layer_norm"],
        }
        if "activation" in self.network_kwargs:
            network_args["activation"] = self.network_kwargs["activation"]
        if self.num_ensembles > 1:
            network = ensemblize(network, self.num_ensembles)
        value_net = network(**network_args)
        self.value_net = value_net

    def __call__(self, observations, actions=None, ref_actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
            ref_actions: Reference actions to condition on (optional). Shape should
                be (N, B, H*A) for sampled chunks, or (B, N*H*A) if already flat.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        if ref_actions is not None:
            ref_actions = flatten_ref_actions(ref_actions, observations.shape[0])
            observations = jnp.concatenate([observations, ref_actions], axis=-1)
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        v = self.value_net(inputs).squeeze(-1)
        return v
