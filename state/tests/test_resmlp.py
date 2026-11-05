"""Regression checks for the official OCBench inner-layer depth convention."""

from pathlib import Path
import json

import flax.linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.networks import ResidualMLP, ResidualMLPBlock, default_init


class EightLayerReferenceMLP(nn.Module):
    """Explicit eight-inner-layer graph, excluding input/output projections."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8, kernel_init=default_init())(x)
        x = nn.LayerNorm()(nn.gelu(x))
        for _ in range(0, 8, 2):
            x = ResidualMLPBlock(8)(x)
        return nn.Dense(3, kernel_init=default_init())(x)


def test_eight_inner_layers_match_reference_parameters_outputs_and_gradients():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5))
    reference, official = EightLayerReferenceMLP(), ResidualMLP((8,) * 8, 3)
    params = reference.init(jax.random.PRNGKey(2), x)
    new_params = official.init(jax.random.PRNGKey(2), x)
    assert jax.tree.structure(params) == jax.tree.structure(new_params)
    for old, new in zip(jax.tree.leaves(params), jax.tree.leaves(new_params)):
        np.testing.assert_array_equal(old, new)
    # Eight inner dense layers plus the input and output projections.
    assert sum(key[-1] == "kernel" for key in flatten_dict(params["params"])) == 10
    np.testing.assert_array_equal(reference.apply(params, x), official.apply(params, x))
    old_grad = jax.grad(lambda value: reference.apply(params, value).sum())(x)
    new_grad = jax.grad(lambda value: official.apply(params, value).sum())(x)
    np.testing.assert_array_equal(old_grad, new_grad)


@pytest.mark.parametrize("dims,message", [((8,) * 3, "even"), ((8, 16), "constant")])
def test_invalid_resmlp_depth_or_width_is_rejected(dims, message):
    with pytest.raises(ValueError, match=message):
        ResidualMLP(dims, 3).init(jax.random.PRNGKey(0), jnp.zeros((2, 5)))


def test_all_selected_ocbench_recipes_keep_four_1024_wide_blocks():
    configs = Path(__file__).resolve().parents[1] / "configs"
    recipes = list(configs.glob("*-cpu-*/*.json"))
    assert len(recipes) == 40
    for path in recipes:
        base = json.loads(path.read_text())["base_policy"]
        assert base["mlp_class"] == "resmlp"
        assert base["hidden_dims"] == [1024] * 8
