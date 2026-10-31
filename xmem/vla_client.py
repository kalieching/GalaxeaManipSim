"""Prompt the G0.5 VLA from a browser and watch it act in GalaxeaManipSim.

Steps the sim with actions from a running GalaxeaVLA policy server
(xmem/serve_policy_5090.py) and serves a web page with a live view, a prompt
box, and Run/Stop buttons. See xmem/README.md.
"""
import argparse
import functools
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gymnasium as gym
import msgpack
import numpy as np
from PIL import Image
from websockets.sync.client import connect

import galaxea_sim.envs  # noqa: F401  (registers envs)

ROBOTS = {
    "r1lite": dict(env="R1LiteCubeBowlPlace-v0", embodiment="galaxea_r1lite"),
    "r1pro": dict(env="R1ProCubeBowlPlace-v0", embodiment="galaxea_r1pro"),
}
GRIPPER_OPEN_M = 0.05  # sim finger joint travel; G0.5 uses 0 (closed) .. 100 (open)
VIEW_HEIGHT = 360


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


class SimVLARunner:
    """Owns the env and the policy connection; all sim calls happen on one thread."""

    def __init__(self, robot: str, server: str, max_steps: int):
        self.cfg = ROBOTS[robot]
        self.server = server
        self.max_steps = max_steps
        self.env = gym.make(self.cfg["env"], headless=True, obs_mode="state")
        self.u = self.env.unwrapped
        self.ws = None
        self.lock = threading.Lock()
        self.frame = None  # latest JPEG bytes
        self.frame_cond = threading.Condition()
        self.status = dict(state="idle", prompt="", step=0, success=False, message="", cot="")
        self.stop_event = threading.Event()
        self.queue = []
        self.wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    # -- requests from the web page --
    def request(self, kind, prompt=""):
        with self.lock:
            self.queue.append((kind, prompt))
        self.wake.set()

    def stop(self):
        self.stop_event.set()

    # -- worker thread --
    def _loop(self):
        self._reset_scene()
        while True:
            self.wake.wait()
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    kind, prompt = self.queue.pop(0)
                self.stop_event.clear()
                try:
                    if kind == "reset":
                        self._reset_scene()
                    elif kind == "run":
                        self._run(prompt)
                except Exception as e:  # keep the UI alive; report the error
                    self._set(state="error", message=f"{type(e).__name__}: {e}")
                    self.ws = None

    def _set(self, **kw):
        with self.lock:
            self.status.update(kw)

    def _reset_scene(self):
        self.env.reset()
        self._set(state="idle", step=0, success=False, message="new scene", cot="")
        self._publish_frame()

    def _connect(self):
        if self.ws is None:
            self.ws = connect(self.server, max_size=None, open_timeout=10)
            unpackb(self.ws.recv())  # server hello: {"action_steps": N}
        return self.ws

    def _run(self, prompt):
        ws = self._connect()
        ws.send(packb({"__reset__": True}))
        unpackb(ws.recv())
        self._set(state="running", prompt=prompt, step=0, success=False, message="", cot="")
        command = self._current_command()
        need_obs = True
        for step in range(1, self.max_steps + 1):
            if self.stop_event.is_set():
                self._set(state="stopped", message=f"stopped at step {step - 1}")
                return
            ws.send(packb(self._observation(prompt) if need_obs else {}))
            resp = unpackb(ws.recv())
            if "error" in resp:
                raise RuntimeError(resp["error"].get("message", resp["error"]))
            need_obs = resp.get("need_obs", True)
            command = self._apply(resp["action"], command)
            info = self.u.step(np.concatenate(
                [command["left_arm"], command["left_gripper"], command["right_arm"], command["right_gripper"]]
            ))[-1]
            if resp.get("cot_text"):
                # keep G0.5's readable reasoning (subtask, bbox, ...), drop raw action tokens
                self._set(cot=re.sub(r"<[^>]*>|\|", " ", resp["cot_text"]).strip())
            self._set(step=step, success=bool(info["success"]))
            self._publish_frame()
            if info["success"]:
                self._set(state="done", message=f"success at step {step}")
                return
        self._set(state="done", message=f"no success after {self.max_steps} steps")

    # -- sim <-> G0.5 conversions --
    def _gripper_state(self, indices):
        return self.u.robot.get_qpos()[indices[0]] / GRIPPER_OPEN_M * 100.0

    def _observation(self, prompt):
        qpos = self.u.robot.get_qpos()
        head, left, right = self.u.cameras
        return dict(
            images=dict(
                head_rgb=self._rgb(head).transpose(2, 0, 1),
                left_wrist_rgb=self._rgb(left).transpose(2, 0, 1),
                right_wrist_rgb=self._rgb(right).transpose(2, 0, 1),
            ),
            state=dict(
                left_arm=qpos[self.u.left_arm_joint_indices].astype(np.float32),
                left_gripper=np.float32([self._gripper_state(self.u.left_gripper_joint_indices)]),
                right_arm=qpos[self.u.right_arm_joint_indices].astype(np.float32),
                right_gripper=np.float32([self._gripper_state(self.u.right_gripper_joint_indices)]),
            ),
            task=prompt,
            embodiment_type=self.cfg["embodiment"],
        )

    def _current_command(self):
        qpos = self.u.robot.get_qpos()
        return dict(
            left_arm=qpos[self.u.left_arm_joint_indices],
            left_gripper=np.array([self.u.last_gripper_cmd[0]]),
            right_arm=qpos[self.u.right_arm_joint_indices],
            right_gripper=np.array([self.u.last_gripper_cmd[1]]),
        )

    def _apply(self, action, command):
        """Overwrite predicted parts; parts G0.5 left out keep their last command."""
        command = dict(command)
        for arm in ("left_arm", "right_arm"):
            if arm in action:
                command[arm] = np.asarray(action[arm], dtype=np.float64).reshape(-1)
        for grip in ("left_gripper", "right_gripper"):
            if grip in action:
                g = float(np.asarray(action[grip]).reshape(-1)[0])
                command[grip] = np.array([np.clip(g / 100.0, 0.0, 1.0) * GRIPPER_OPEN_M])
        return command

    # -- rendering --
    @staticmethod
    def _rgb(camera):
        camera.take_picture()
        return (camera.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)

    def _publish_frame(self):
        views = [self._rgb(self.u.default_camera), self._rgb(self.u.cameras[0])]
        views = [Image.fromarray(v).resize((int(v.shape[1] * VIEW_HEIGHT / v.shape[0]), VIEW_HEIGHT)) for v in views]
        canvas = Image.new("RGB", (sum(v.width for v in views), VIEW_HEIGHT))
        x = 0
        for v in views:
            canvas.paste(v, (x, 0))
            x += v.width
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=80)
        with self.frame_cond:
            self.frame = buf.getvalue()
            self.frame_cond.notify_all()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>xmem VLA sim</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}
img{max-width:100%;border:1px solid #333}
form{display:flex;gap:8px;margin:12px 0}
input{flex:1;padding:8px;font-size:15px;background:#222;color:#eee;border:1px solid #444}
button{padding:8px 14px;font-size:15px}
pre{white-space:pre-wrap;background:#1b1b1b;padding:8px;font-size:12px;max-height:8em;overflow:auto}
</style></head><body>
<h3>__TITLE__</h3>
<img src="/stream" alt="sim view (scene camera | head camera)">
<form onsubmit="post('/run');return false">
<input id="p" value="__PROMPT__"><button>Run</button>
<button type="button" onclick="post('/stop')">Stop</button>
<button type="button" onclick="post('/reset')">New scene</button></form>
<div id="s"></div><pre id="c"></pre>
<script>
function post(u){fetch(u,{method:'POST',body:document.getElementById('p').value})}
setInterval(async()=>{const s=await (await fetch('/status')).json();
document.getElementById('s').textContent=`${s.state} | step ${s.step} | success ${s.success} | ${s.message}`;
document.getElementById('c').textContent=s.cot||''},500);
</script></body></html>"""


def make_handler(runner: SimVLARunner, title: str):
    page = PAGE.replace("__TITLE__", title).replace("__PROMPT__", runner.u.language_instruction).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/status":
                with runner.lock:
                    self._send(json.dumps(runner.status).encode(), "application/json")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        # Browsers draw a multipart frame only once the next one arrives,
                        # so resend the latest frame when idle instead of waiting for a new one.
                        with runner.frame_cond:
                            runner.frame_cond.wait_for(lambda: runner.frame is not last, timeout=0.5)
                            frame = last = runner.frame
                        if frame is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode().strip()
            if self.path == "/run" and body:
                runner.request("run", body)
            elif self.path == "/reset":
                runner.request("reset")
            elif self.path == "/stop":
                runner.stop()
            else:
                return self.send_error(400)
            self._send(b"ok", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", choices=ROBOTS, default="r1lite")
    ap.add_argument("--server", default="ws://127.0.0.1:8765", help="policy server URL")
    ap.add_argument("--port", type=int, default=8000, help="web page port (bound to 127.0.0.1)")
    ap.add_argument("--max-steps", type=int, default=900, help="steps per prompt (15 Hz)")
    args = ap.parse_args()

    runner = SimVLARunner(args.robot, args.server, args.max_steps)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner, f"{runner.cfg['env']} ← G0.5"))
    print(f"Open http://127.0.0.1:{args.port} (policy server: {args.server})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
