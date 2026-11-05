import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def supply_rng(f, rng=None):
    """
    Wraps a function to supply jax rng. It will remember the rng state for that function.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def cosine_warm_restarts(
    peak_lr, min_lr, first_cycle_steps, warmup_steps, cycle_gamma=1.0
):
    """DICE-RL's CosineAnnealingWarmupRestarts (cycle_mult=1) as an optax schedule.

    ``count`` is the number of updates already applied by the optimizer.  Every
    cycle of ``first_cycle_steps`` updates starts with ``warmup_steps`` of linear
    warmup from ``min_lr`` to the cycle's peak, then cosine-decays back to
    ``min_lr``; the peak is multiplied by ``cycle_gamma`` after each cycle.
    """

    def schedule(count):
        cycle = count // first_cycle_steps
        step_in_cycle = count % first_cycle_steps
        peak = peak_lr * cycle_gamma**cycle
        warmup = min_lr + (peak - min_lr) * step_in_cycle / warmup_steps
        progress = (step_in_cycle - warmup_steps) / (first_cycle_steps - warmup_steps)
        cosine = min_lr + (peak - min_lr) * (1 + jnp.cos(jnp.pi * progress)) / 2
        return jnp.where(step_in_cycle < warmup_steps, warmup, cosine)

    return schedule


def module_mask(module_name):
    """Prefix mask selecting one ``ModuleDict`` module's parameters (for weight decay)."""
    return lambda params: {key: key == f"modules_{module_name}" for key in params}


def make_optimizer(
    lr,
    *,
    max_grad_norm=None,
    weight_decay=0.0,
    weight_decay_mask=None,
    lr_schedule="none",
    lr_first_cycle_steps=1000,
    lr_warmup_steps=10,
    lr_min=1e-06,
    lr_cycle_gamma=1.0,
):
    """Adam with DICE-RL's optional optimizer tricks.

    The chain is global-norm clipping -> coupled L2 weight decay (added to the
    gradient before Adam, i.e. ``torch.optim.Adam(weight_decay=...)``) -> Adam
    -> learning rate (constant or ``cosine_warm_restarts``).  With the defaults
    it is exactly ``optax.adam(lr)``: same update and same optimizer state, so
    checkpoints saved with ``optax.adam`` restore unchanged.  Pass
    ``weight_decay_mask=module_mask(name)`` when one optimizer per loss spans
    every module's parameters, so only the loss's own module is decayed.  The
    schedule count is this optimizer's own update count, so with per-loss
    optimizers a skipped loss also pauses its schedule.
    """
    parts = []
    if max_grad_norm is not None:
        parts.append(optax.clip_by_global_norm(max_grad_norm))
    if weight_decay > 0:
        parts.append(optax.add_decayed_weights(weight_decay, mask=weight_decay_mask))
    parts.append(optax.scale_by_adam())
    if lr_schedule == "cosine_warm_restarts":
        schedule = cosine_warm_restarts(
            lr, lr_min, lr_first_cycle_steps, lr_warmup_steps, lr_cycle_gamma
        )
        parts.append(optax.scale_by_schedule(schedule))
        parts.append(optax.scale(-1.0))
    else:
        assert lr_schedule == "none", lr_schedule
        parts.append(optax.scale(-lr))
    return optax.chain(*parts)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @staticmethod
    def _is_tx(x):
        return isinstance(x, optax.GradientTransformation)

    @classmethod
    def _tx_tree_map(cls, *args, **kwargs):
        return jax.tree_util.tree_map(*args, is_leaf=cls._is_tx, **kwargs)

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            if isinstance(tx, Mapping):
                opt_state = cls._tx_tree_map(lambda tx_: tx_.init(params), tx)
            else:
                opt_state = tx.init(params)
        else:
            opt_state = None
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {"params": params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state.

        With a per-loss ``tx`` mapping, only the losses present in ``grads``
        step their optimizer; the others keep their state (Adam moments and
        schedule count), so a skipped loss (e.g. a delayed actor update) leaves
        its module untouched instead of drifting on stale momentum.
        """
        if isinstance(self.tx, Mapping):
            new_opt_state = dict(self.opt_state)
            updates = jax.tree_util.tree_map(jnp.zeros_like, self.params)
            for name, grad in grads.items():
                update, new_opt_state[name] = self.tx[name].update(
                    grad, self.opt_state[name], self.params
                )
                updates = jax.tree_util.tree_map(jnp.add, updates, update)
        else:
            updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)
        grad_max_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0
        )
        grad_min_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0
        )
        grad_norm_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0
        )
        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)
        info.update(
            {
                "grad/max": final_grad_max,
                "grad/min": final_grad_min,
                "grad/norm": final_grad_norm,
            }
        )
        return (self.apply_gradients(grads=grads), info)

    def apply_loss_fns(self, loss_fns, rng=None, has_aux=False, loss_terms=None):
        """Apply losses; optionally log global L2 parameter-gradient norms.

        ``loss_terms`` maps loss names to weighted scalar terms in their aux
        dictionaries. Norms include all parameter leaves, before optimization.
        The VJP reuses the exact forward pass and random draws for the total
        gradient and every term; diagnostic gradients never update parameters.
        """
        treedef = jax.tree_util.tree_structure(loss_fns)
        rng = rng if rng is not None else jax.random.PRNGKey(self.step)
        rngs = jax.random.split(rng, treedef.num_leaves)
        rngs = jax.tree_util.tree_unflatten(treedef, rngs)
        if loss_terms is None:
            grads_and_aux = jax.tree_util.tree_map(
                lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(
                    self.params, rng
                ),
                loss_fns,
                rngs,
            )
        else:
            assert has_aux
            grads_and_aux = {}
            for name, loss_fn in loss_fns.items():
                if name not in loss_terms:
                    grads_and_aux[name] = jax.grad(loss_fn, has_aux=True)(
                        self.params, rngs[name]
                    )
                    continue
                term_names = loss_terms[name]

                def terms_fn(params):
                    total, info = loss_fn(params, rngs[name])
                    return (
                        jnp.stack([total] + [info[key] for key in term_names]),
                        info,
                    )

                values, pullback, info = jax.vjp(terms_fn, self.params, has_aux=True)
                cotangent = jnp.zeros_like(values).at[0].set(1.0)
                grads = pullback(cotangent)[0]
                info[f"{name}/grad_norm/total"] = optax.global_norm(grads)
                for index, key in enumerate(term_names, start=1):
                    term_grads = pullback(jnp.zeros_like(values).at[index].set(1.0))[0]
                    info[f"{name}/grad_norm/{key.split('/')[-1]}"] = optax.global_norm(
                        term_grads
                    )
                grads_and_aux[name] = (grads, info)
        if has_aux:
            grads = jax.tree_util.tree_map(
                lambda _, value: value[0], loss_fns, grads_and_aux
            )
            aux = jax.tree_util.tree_map(
                lambda _, value: value[1], loss_fns, grads_and_aux
            )
            return (self.apply_gradients(grads=grads), aux)
        return self.apply_gradients(grads=grads_and_aux)


def atomic_pickle_dump(obj, save_path, max_retries=3, retry_delay=30):
    """Pickle ``obj`` to ``save_path`` via a temp file and rename, retrying on I/O errors.

    Scratch (Lustre) occasionally fails large writes with ``[Errno 5]``; writing
    to ``<save_path>.tmp`` first means a failed attempt never leaves a truncated
    checkpoint behind, and a later auto-resume only ever sees complete files.
    """
    import time

    tmp_path = save_path + ".tmp"
    for attempt in range(max_retries):
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(obj, f)
            os.replace(tmp_path, save_path)
            return
        except (OSError, IOError) as e:
            print(
                f"Save attempt {attempt + 1}/{max_retries} for {save_path} failed: {e}"
            )
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def save_agent(agent, save_dir, epoch):
    """Save the agent to ``<save_dir>/params_<epoch>.pkl``."""
    save_path = os.path.join(save_dir, f"params_{epoch}.pkl")
    atomic_pickle_dump(dict(agent=flax.serialization.to_state_dict(agent)), save_path)
    print(f"Saved to {save_path}")


def _resolve_checkpoint_dir_from_glob(restore_path_glob: str) -> str:
    """Resolve a checkpoint directory; may be a glob pattern.

    If multiple directories match, picks the first after lexicographic sort so
    choice is stable across machines and runs.
    """
    candidates = glob.glob(restore_path_glob)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directory matched pattern: {restore_path_glob!r}"
        )
    if len(candidates) > 1:
        chosen = sorted(candidates)[0]
        print(
            f"Warning: {len(candidates)} dirs matched {restore_path_glob!r}; using first after sort: {chosen}"
        )
        return chosen
    return candidates[0]


def _check_checkpoint_parameter_shapes(
    expected, restored, path="agent", in_params=False
):
    """Reject incompatible observation/network shapes before replacing weights."""
    if isinstance(expected, Mapping) and isinstance(restored, Mapping):
        for key in expected.keys() & restored.keys():
            _check_checkpoint_parameter_shapes(
                expected[key],
                restored[key],
                f"{path}.{key}",
                in_params=in_params or key == "params",
            )
    elif in_params and hasattr(expected, "shape") and hasattr(restored, "shape"):
        if expected.shape != restored.shape:
            raise ValueError(
                f"Checkpoint parameter shape mismatch at {path}: checkpoint {restored.shape}, expected {expected.shape}. Use a checkpoint trained with matching observations and network architecture. Legacy 50D Transport checkpoints do not match the corrected 59D two-robot observations; retrain the base policy and downstream agent with the corrected inputs."
            )


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    restore_path = (
        _resolve_checkpoint_dir_from_glob(restore_path) + f"/params_{restore_epoch}.pkl"
    )
    with open(restore_path, "rb") as f:
        load_dict = pickle.load(f)
    agent_state = load_dict["agent"]
    if isinstance(agent_state, dict):
        agent_state.pop("qgrad_ema_m1", None)
        agent_state.pop("qgrad_ema_m2", None)
    if (
        isinstance(agent_state, dict)
        and hasattr(agent, "one_step_actor")
        and ("one_step_actor" not in agent_state)
    ):
        if agent.one_step_actor is None:
            agent_state = {**agent_state, "one_step_actor": None}
        else:
            agent_state = {
                **agent_state,
                "one_step_actor": flax.serialization.to_state_dict(
                    agent.one_step_actor
                ),
            }
            print(
                "Warning: checkpoint has no one_step_actor state; keeping the current (freshly initialized) one_step_actor weights."
            )
    if (
        isinstance(agent_state, dict)
        and hasattr(agent, "one_step_actor")
        and (agent_state.get("one_step_actor") is None)
        and (agent.one_step_actor is not None)
    ):
        agent_state = {
            **agent_state,
            "one_step_actor": flax.serialization.to_state_dict(agent.one_step_actor),
        }
        print(
            "Warning: checkpoint has one_step_actor=None; keeping the current (freshly initialized) one_step_actor weights."
        )
    _check_checkpoint_parameter_shapes(
        flax.serialization.to_state_dict(agent), agent_state
    )
    agent = flax.serialization.from_state_dict(agent, agent_state)
    print(f"Restored from {restore_path}")
    return agent
