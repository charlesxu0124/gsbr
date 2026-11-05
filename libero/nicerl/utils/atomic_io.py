"""Crash-safe pickling for checkpoints that a spot node may lose mid-write.

`atomic_pickle` writes to a temp file, fsyncs, keeps the previous version as
``<path>.prev`` and renames into place, so a reader never sees a truncated file.
`load_pickle` restores ``<path>`` and falls back to ``<path>.prev`` when the
newest copy is unreadable (killed during the rename window, or a mirror that
copied a stale pair).
"""

import os
import pickle


def atomic_pickle(obj, path: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, f"{path}.prev")
    os.replace(tmp, path)


def load_pickle(path: str):
    """Return the unpickled object at ``path`` or its ``.prev`` fallback; raise if neither loads."""
    errors = []
    for candidate in (path, f"{path}.prev"):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:  # noqa: BLE001  truncated / corrupt file
            errors.append(f"{candidate}: {type(e).__name__}: {e}")
            continue
        if candidate != path:
            print(
                f"[atomic_io] {path} unreadable ({errors[-1] if errors else 'missing'}); using {candidate}"
            )
        return obj
    raise FileNotFoundError(
        f"no readable checkpoint at {path} (tried .prev too): {errors}"
    )
