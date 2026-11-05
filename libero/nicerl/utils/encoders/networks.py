from typing import Any, Dict, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

# He init (variance 2/fan_in): preserves forward variance through the ReLU stack.
default_init = nn.initializers.he_uniform


class PixelStateEncoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = "VALID"

    use_bottleneck: bool = False
    bottleneck_dim: Optional[int] = None

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations["pixels"].astype(jnp.float32) / 255.0  # [B, H, W, C, NCAMS]
        x = jnp.reshape(x, (*x.shape[:-2], -1))  # [B, H, W, C*NCAMS]

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(
                features,
                kernel_size=(3, 3),
                strides=(stride, stride),
                kernel_init=default_init(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)

        # [B, H', W', feature_dim] --> [B, H'*W'*feature_dim]
        x = x.reshape((*x.shape[:-3], -1))

        # [B, H'*W'*feature_dim] --> [B, bottleneck_dim]
        if self.use_bottleneck:
            assert self.bottleneck_dim is not None
            x = nn.Dense(self.bottleneck_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        if "state" in observations:
            return jnp.concatenate([x, observations["state"]], axis=-1)
        return x


class ObservationActionMultiplexer(nn.Module):
    network: nn.Module
    obs_encoder: Optional[nn.Module] = None
    action_encoder: Optional[nn.Module] = None
    do_combine_encoding: bool = False

    @nn.compact
    def __call__(
        self, observations: Any, actions: Optional[Any] = None, **kwargs
    ) -> jnp.ndarray:
        """Multiplex observations and actions into a single vector.

        Args:
            observations: Observations (dict or array).
            actions: Actions (optional).
        """

        # Obs --> obs_features
        if self.obs_encoder is not None:
            obs_features = self.obs_encoder(observations)
        else:
            obs_features = observations

        # Actions --> action_features
        if actions is not None:
            if self.action_encoder is not None:
                action_features = self.action_encoder(actions)
            else:
                action_features = actions
        else:
            action_features = None

        # K candidate actions per observation, (B, K, A): encode the observation
        # once and broadcast its features over the candidates, so a critic scores
        # (B, K) actions with a single encoder pass and returns (E, B, K).
        if (
            action_features is not None
            and not isinstance(obs_features, dict)
            and action_features.ndim == obs_features.ndim + 1
        ):
            obs_features = jnp.broadcast_to(
                obs_features[:, None],
                action_features.shape[:-1] + obs_features.shape[-1:],
            )

        # Pass now multiplexed features to the network.
        if self.do_combine_encoding:
            if action_features is not None:
                inputs = jnp.concatenate([obs_features, action_features], axis=-1)
            else:
                inputs = obs_features
            return self.network(inputs, **kwargs)
        else:
            return self.network(obs_features, action_features, **kwargs)
