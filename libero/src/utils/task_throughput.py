"""Task throughput: successful episodes per simulated hour, not compute speed."""


def episode_env_steps(env, chunks):
    # LIBERO counts actual primitive actions, including a partial final chunk;
    # current_step is reset after the ten simulator stabilization actions.
    inner = getattr(env, "env", None)
    return int(getattr(inner, "current_step", chunks))


def control_frequency(env, fallback):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        hz = getattr(current, "control_freq", None)
        if hz is not None:
            return float(hz)
        current = getattr(current, "env", None)
    return float(fallback)


def task_throughput(successes, lengths, hz):
    """3600 * control Hz * mean(success) / mean(all episode lengths)."""
    if len(successes) != len(lengths) or not lengths:
        raise ValueError("Need paired, nonempty success and length samples")
    if hz <= 0 or any(length <= 0 for length in lengths):
        raise ValueError("Control frequency and episode lengths must be positive")
    return 3600.0 * hz * sum(successes) / sum(lengths)
