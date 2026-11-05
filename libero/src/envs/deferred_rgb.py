"""Defer standard LIBERO RGB rendering until an action chunk returns.

Keep the observable clocks unchanged and copy full MuJoCo data at each original
camera sampling time. Render the last sample, including when success or timeout
ends a chunk early. Only the render context reads the copy; physics always uses
the original live data. This relies on robosuite's native MuJoCo render context.
"""

from functools import wraps
from types import SimpleNamespace

import mujoco
import numpy as np
from robosuite.utils.observables import NO_CORRUPTION, NO_DELAY, NO_FILTER


class DeferredRGBRenderer:
    def __init__(self, env):
        self.env = env
        self.base = env.env.env.env
        self.sim = self.base.sim
        self.cameras = tuple(f"{name}_image" for name in env.env.CAMERA_NAMES)
        self.snapshots = {}
        self.sensor_calls = 0
        self.actual_renders = 0

    def _supported(self):
        base = self.base
        context = self.sim._render_context_offscreen
        if (
            not hasattr(mujoco, "mj_copyData")
            or context is None
            or context.data is not self.sim.data
            or any(base.camera_depths)
            or any(s is not None for s in base.camera_segmentations)
            # These scenes mutate the model's visual properties after the
            # sensor sample (e.g. stove lights); MjData alone cannot capture it.
            or getattr(base, "visualization_sites_list", ())
            or getattr(base, "tracking_object_states_change", ())
        ):
            return False
        images = {
            name for name, obs in base._observables.items() if obs.modality == "image"
        }
        if images != set(self.cameras):
            return False
        return all(
            obs._corrupter is NO_CORRUPTION
            and obs._filter is NO_FILTER
            and obs._delayer is NO_DELAY
            and obs.is_enabled()
            and obs.is_active()
            for obs in (base._observables[name] for name in self.cameras)
        )

    def step(self, original, actions):
        # These callers consume intermediate images or reset inside the chunk.
        # Preserve their original path, as well as unsupported camera pipelines.
        if (
            self.env.auto_reset
            or self.env.info_captures_all_frames
            or len(actions) <= 1
            or not self._supported()
        ):
            return original(actions)
        if not self.snapshots:
            self.snapshots = {
                name: mujoco.MjData(self.sim.model._model) for name in self.cameras
            }
        observables = self.base._observables
        originals = {name: observables[name]._sensor for name in self.cameras}
        sampled = set()
        try:
            for name in self.cameras:
                observable = observables[name]

                @wraps(originals[name])
                def capture(cache, name=name, observable=observable):
                    mujoco.mj_copyData(
                        self.snapshots[name], self.sim.model._model, self.sim.data._data
                    )
                    sampled.add(name)
                    self.sensor_calls += 1
                    return observable._current_observed_value

                observable._sensor = capture
            result = original(actions)
        finally:
            for name, sensor in originals.items():
                observables[name]._sensor = sensor

        obs, reward, terminated, truncated, info = result
        context = self.sim._render_context_offscreen
        live_data = context.data
        try:
            for name in self.cameras:
                if name not in sampled:
                    continue
                context.data = SimpleNamespace(_data=self.snapshots[name])
                frame = np.array(originals[name](self.base._obs_cache))
                observables[name]._current_observed_value = frame
                self.base._obs_cache[name] = frame.copy()
                # Match LiberoEnv._process_obs, including both image flips.
                obs[name] = frame[::-1, ::-1].copy()
                self.actual_renders += 1
        finally:
            context.data = live_data
        return obs, reward, terminated, truncated, info
