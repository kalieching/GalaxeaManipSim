"""Drive a GalaxeaManipSim env with a running GalaxeaVLA G0.5 policy server (msgpack over WebSocket).

The server serves one action per call from a cached chunk and asks for a new observation (camera images and joint
state) when the chunk runs out. Parts of the robot that G0.5 leaves out of an action keep their last command.
"""
import functools

import msgpack
import numpy as np
from websockets.sync.client import connect

GRIPPER_OPEN_M = 0.05  # sim finger joint travel (G0.5 uses 0 closed .. 100 open)
PARTS = ("left_arm", "left_gripper", "right_arm", "right_gripper")


# ---- msgpack + numpy wire format (same as GalaxeaVLA src/g05/utils/websocket/msgpack.py) ----
def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tobytes(), "dtype": obj.dtype.str, "shape": obj.shape}
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    obj = {(k.decode() if isinstance(k, bytes) else k): v for k, v in obj.items()}
    if "__ndarray__" in obj:
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=tuple(obj["shape"]))
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)


def rgb(camera):
    camera.take_picture()
    return (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)


class G05Policy:
    def __init__(self, server, embodiment):
        self.server, self.embodiment = server, embodiment
        self.ws, self.need_obs, self.command = None, True, None

    def reset(self, env):
        """Start a new episode on the server. The robot's current pose and gripper commands are the hold command."""
        if self.ws is None:
            self.ws = connect(self.server, max_size=None, open_timeout=10)
            unpackb(self.ws.recv())  # server hello: {"action_steps": N}
        self.ws.send(packb({"__reset__": True}))
        unpackb(self.ws.recv())
        self.need_obs = True
        qpos = env.robot.get_qpos()
        self.command = dict(left_arm=qpos[env.left_arm_joint_indices], left_gripper=np.array([env.last_gripper_cmd[0]]),
                            right_arm=qpos[env.right_arm_joint_indices], right_gripper=np.array([env.last_gripper_cmd[1]]))

    def observation(self, env, prompt):
        qpos = env.robot.get_qpos()
        head, left, right = env.cameras
        grip = lambda idx: np.float32([qpos[idx[0]] / GRIPPER_OPEN_M * 100.0])  # noqa: E731
        return dict(
            images=dict(head_rgb=rgb(head).transpose(2, 0, 1), left_wrist_rgb=rgb(left).transpose(2, 0, 1),
                        right_wrist_rgb=rgb(right).transpose(2, 0, 1)),
            state=dict(left_arm=qpos[env.left_arm_joint_indices].astype(np.float32),
                       left_gripper=grip(env.left_gripper_joint_indices),
                       right_arm=qpos[env.right_arm_joint_indices].astype(np.float32),
                       right_gripper=grip(env.right_gripper_joint_indices)),
            task=prompt,
            embodiment_type=self.embodiment,
        )

    def act(self, env, prompt):
        """One control step. Returns (env action, observation sent to G0.5 or None, G0.5's raw action dict)."""
        obs = self.observation(env, prompt) if self.need_obs else None
        self.ws.send(packb(obs or {}))
        resp = unpackb(self.ws.recv())
        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", resp["error"]))
        self.need_obs = resp.get("need_obs", True)
        action = resp["action"]
        for arm in ("left_arm", "right_arm"):
            if arm in action:
                self.command[arm] = np.asarray(action[arm], dtype=np.float64).reshape(-1)
        for grip in ("left_gripper", "right_gripper"):
            if grip in action:
                g = float(np.asarray(action[grip]).reshape(-1)[0])
                self.command[grip] = np.array([np.clip(g / 100.0, 0.0, 1.0) * GRIPPER_OPEN_M])
        return np.concatenate([self.command[p] for p in PARTS]), obs, action
