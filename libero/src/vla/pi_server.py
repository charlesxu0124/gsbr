import os
import json_numpy
import json
import logging
import threading
import time
import traceback
from typing import Any, Dict
import argparse
import inspect
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
from copy import deepcopy
from typing import Optional, Union
import math
from openpi.training import config as openpi_config
from openpi.models import model as openpi_model
from openpi.policies import policy_config
from openpi.shared import download
from openpi_client import image_tools
from src.vla.pi_shared_prefix import make_shared_prefix_sampler


class LiberoObsWrangler:
    @staticmethod
    def quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """
        Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
        """
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return quat[:3] * 2.0 * math.acos(quat[3]) / den

    @staticmethod
    def obs_to_qpos(obs: Dict[str, np.ndarray]) -> np.ndarray:
        obs_copy = deepcopy(obs)
        qpos = np.concatenate(
            (
                obs_copy["robot0_eef_pos"],
                LiberoObsWrangler.quat2axisangle(obs_copy["robot0_eef_quat"]),
                obs_copy["robot0_gripper_qpos"],
            )
        )
        return qpos

    @staticmethod
    def preprocess_pi0_obs(
        obs: Dict[str, np.ndarray],
    ) -> Dict[str, Union[np.ndarray, None, str]]:
        obs_copy = deepcopy(obs)
        img = np.ascontiguousarray(obs_copy["agentview_image"])
        wrist_img = np.ascontiguousarray(obs_copy["robot0_eye_in_hand_image"])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs_copy["robot0_eef_pos"],
                    LiberoObsWrangler.quat2axisangle(obs_copy["robot0_eef_quat"]),
                    obs_copy["robot0_gripper_qpos"],
                )
            ),
            "prompt": obs_copy["task_language"],
        }


class PIServer:
    def __init__(
        self, policy_name: str, env_name: str, checkpoint_path: Optional[str] = None
    ):
        self.policy_name = policy_name
        self.env_name = env_name
        if policy_name != "pi05_libero":
            raise ValueError("Only pi05_libero is supported")
        config = openpi_config.get_config("pi05_libero")
        checkpoint_dir = download.maybe_download(
            checkpoint_path or "gs://openpi-assets/checkpoints/pi05_libero"
        )
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)
        if env_name != "libero":
            raise ValueError("Only LIBERO is supported")
        self.obs_preprocess_fn = LiberoObsWrangler.preprocess_pi0_obs
        self._batch_sampler = make_shared_prefix_sampler(self.policy._model)
        self._inference_lock = threading.Lock()
        infer_parameters = inspect.signature(self.policy.infer).parameters
        self.supports_sampling_window = {"num_steps", "start_step"}.issubset(
            infer_parameters
        )

    def _predict_action_batch(
        self,
        obs: Dict[str, Any],
        noise: Optional[np.ndarray],
        batch_size: int,
        num_steps: int,
        start_step: int,
    ) -> Dict[str, Any]:
        """Sample independent action chunks with one shared observation prefill.

        OpenPI's public ``Policy.infer`` always inserts a singleton batch and
        removes it afterward. GSBR needs many independent diffusion samples for
        the same observation, so this uses the pinned Policy's preprocessing
        and model sampler directly while preserving its output transforms.
        """
        required_attributes = (
            "_input_transform",
            "_output_transform",
            "_sample_actions",
            "_sample_kwargs",
            "_is_pytorch_model",
        )
        missing = [
            name for name in required_attributes if not hasattr(self.policy, name)
        ]
        if missing:
            raise RuntimeError(
                f"This OpenPI Policy does not expose the native batch interface; missing {missing}"
            )
        inputs = jax.tree.map(lambda value: value, obs)
        inputs = self.policy._input_transform(inputs)
        sample_kwargs = dict(self.policy._sample_kwargs)

        def to_batched_array(value):
            value = jnp.asarray(value)
            input_batch = 1
            return jnp.broadcast_to(value, (input_batch, *value.shape))

        inputs = jax.tree.map(to_batched_array, inputs)
        self.policy._rng, sample_rng_or_device = jax.random.split(self.policy._rng)
        if noise is not None:
            if noise.ndim != 3 or noise.shape[0] != batch_size:
                raise ValueError(
                    f"batched noise must have shape (batch, horizon, dim); received {noise.shape} for batch_size={batch_size}"
                )
            sample_kwargs["noise"] = jnp.asarray(noise)
        if self.supports_sampling_window:
            sample_kwargs.update(num_steps=num_steps, start_step=start_step)
        elif (num_steps, start_step) != (10, 0):
            raise ValueError(
                f"This OpenPI checkout does not support custom diffusion windows (requested num_steps={num_steps}, start_step={start_step})."
            )
        observation = openpi_model.Observation.from_dict(inputs)
        if (num_steps, start_step) != (10, 0):
            raise ValueError("The release uses the final 10-step pi0.5 schedule")
        if noise is None:
            sample_kwargs["noise"] = jax.random.normal(
                sample_rng_or_device,
                (
                    batch_size,
                    self.policy._model.action_horizon,
                    self.policy._model.action_dim,
                ),
            )
        sample_kwargs.pop("start_step", None)
        sampler = self._batch_sampler
        start_time = time.monotonic()
        actions = sampler(sample_rng_or_device, observation, **sample_kwargs)
        actions = np.asarray(actions)
        states = np.broadcast_to(
            np.asarray(inputs["state"]), (batch_size, *inputs["state"].shape[1:])
        )
        model_time = time.monotonic() - start_time
        transformed_actions = []
        for index in range(batch_size):
            output = self.policy._output_transform(
                {"state": states[index], "actions": actions[index]}
            )
            transformed_actions.append(np.asarray(output["actions"]))
        return {
            "actions": np.stack(transformed_actions),
            "policy_timing": {"infer_ms": model_time * 1000},
        }

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            obs, policy_name, noise, meta = (
                json_numpy.loads(payload["obs"]),
                payload["policy_name"],
                json_numpy.loads(payload["noise"])
                if payload.get("noise") is not None
                else None,
                json_numpy.loads(payload["meta"])
                if payload.get("meta") is not None
                else None,
            )
            assert policy_name == self.policy_name, (
                f"Received request for policy {policy_name}, but server is running policy {self.policy_name}"
            )
            obs = self.obs_preprocess_fn(obs)
            print(f"[PI SERVER] Language instruction: {obs['prompt']}")
            img = obs["observation/image"].copy()
            num_steps = meta.get("num_steps", 10) if meta is not None else 10
            start_step = meta.get("start_step", 0) if meta is not None else 0
            requested_batch_size = payload.get("batch_size")
            with self._inference_lock:
                if requested_batch_size is not None:
                    batch_size = int(requested_batch_size)
                    if batch_size < 1:
                        raise ValueError("batch_size must be positive")
                    output = self._predict_action_batch(
                        obs,
                        noise=noise,
                        batch_size=batch_size,
                        num_steps=num_steps,
                        start_step=start_step,
                    )
                else:
                    infer_kwargs = {"noise": noise}
                    if self.supports_sampling_window:
                        infer_kwargs.update(num_steps=num_steps, start_step=start_step)
                    elif (num_steps, start_step) != (10, 0):
                        raise ValueError(
                            f"This OpenPI checkout does not support custom diffusion windows (requested num_steps={num_steps}, start_step={start_step})."
                        )
                    output = self.policy.infer(obs, **infer_kwargs)
            action_chunk = output["actions"]
            print(
                f"[PI SERVER] batch_size={requested_batch_size or 1} infer_ms={output.get('policy_timing', {}).get('infer_ms', float('nan')):.1f}"
            )
            return JSONResponse(json_numpy.dumps({"action": action_chunk}))
        except:
            logging.error(traceback.format_exc())
            logging.warning("Your request threw an error.")
            return JSONResponse({"error": traceback.format_exc()}, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("src", "cfgs", "pi_server.yaml"),
        help="Path to config file",
    )
    args, hydra_args = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if hydra_args:
        conf_overrides = OmegaConf.from_dotlist(hydra_args)
        cfg = OmegaConf.merge(cfg, conf_overrides)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    server = PIServer(
        policy_name=cfg.policy_name,
        env_name=cfg.env_name,
        checkpoint_path=cfg.get("checkpoint_path"),
    )
    server.run(host=cfg.host, port=cfg.port)
