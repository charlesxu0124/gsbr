from unittest.mock import patch

import json_numpy
import numpy as np

from src.vla.pi_client import PIClient


class _Response:
    status_code = 200

    def __init__(self, actions):
        self.actions = actions

    def raise_for_status(self):
        pass

    def json(self):
        return json_numpy.dumps({"action": self.actions})


def test_pi_client_sends_multiple_samples_as_one_batch_request():
    client = PIClient(policy_name="pi05_libero", host_ip="127.0.0.1")
    obs = {
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "task_language": "ignored by the explicit instruction",
    }
    noise = np.zeros((16, 10, 32), dtype=np.float32)
    response_actions = np.zeros((16, 10, 7), dtype=np.float32)

    with patch(
        "src.vla.pi_client.requests.post",
        return_value=_Response(response_actions),
    ) as post:
        actions = client.sample_actions(
            obs,
            "Do the thing.",
            num_samples=16,
            noise=noise,
            meta={"num_steps": 10, "start_step": 0},
        )

    post.assert_called_once()
    payload = post.call_args.kwargs["json"]
    assert payload["batch_size"] == 16
    assert json_numpy.loads(payload["noise"]).shape == (16, 10, 32)
    assert json_numpy.loads(payload["obs"])["task_language"] == "do the thing"
    assert actions.shape == (16, 10, 7)
    assert actions.dtype == np.float32
