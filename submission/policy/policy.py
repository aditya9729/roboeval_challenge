"""RoboEval entry policy: pi0.5 served over websocket (localhost).

The evaluator calls ``make_policy(checkpoint_dir, metadata)``; we connect to a
``serve_pi05.py`` server described by ``checkpoint_dir/pi05_server.json`` and
fail loudly if its weights hash differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

_POLICY_DIR = Path(__file__).resolve().parent
if str(_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLICY_DIR))

from roboeval_entry.contracts.action_spec import load_action_spec  # noqa: E402
from roboeval_entry.control.guard import ActionGuard  # noqa: E402
from roboeval_entry.policies.base import ObservationSchema  # noqa: E402
from roboeval_entry.policies.official_bridge import OfficialPolicyBridge  # noqa: E402
from roboeval_entry.policies.pi05_client import (  # noqa: E402
    Pi05RemoteAdapter,
    RelaxedWebsocketClient,
)


def make_policy(checkpoint_dir: Path, metadata: Dict[str, Any], client=None):
    info = json.loads((Path(checkpoint_dir) / "pi05_server.json").read_text())
    if client is None:
        client = RelaxedWebsocketClient(host=info["host"], port=info["port"])
    server_sha = client.get_server_metadata().get("model_sha256")
    if server_sha != info["model_sha256"]:
        raise RuntimeError(
            f"model hash mismatch: server {server_sha} != "
            f"staged {info['model_sha256']}")
    cams = metadata["interface"]["observation"]["cameras"]
    cameras = tuple(c["name"] for c in cams)
    rule = None
    if info.get("gripper_rule") is not None:
        from roboeval_entry.control.gripper_rule import GripperCorrectionRule

        rule = GripperCorrectionRule(**info["gripper_rule"])
    adapter = Pi05RemoteAdapter(
        host=info["host"], port=info["port"],
        execute_steps=info["execute_steps"], cameras=cameras, client=client,
        gripper_rule=rule, inpaint_steps=info.get("inpaint_steps", 0),
        state_mode=info.get("state_mode", "qpos18"))
    guard = ActionGuard(load_action_spec(_POLICY_DIR / "action_spec.json"))
    return OfficialPolicyBridge(adapter, ObservationSchema(cameras), guard)
