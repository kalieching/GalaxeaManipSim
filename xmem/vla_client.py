"""Prompt the G0.5 VLA from a browser and watch it act in GalaxeaManipSim.

Steps the sim with actions from a running GalaxeaVLA policy server
(xmem/serve_policy_5090.py) and serves a web page with a live view, a prompt
box, and Run/Stop buttons. See xmem/README.md.
"""
import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import galaxea_sim.envs  # noqa: F401  (registers envs)
from policy_client import G05Policy, rgb

ROBOTS = {
    "r1lite": dict(env="R1LiteCubeBowlPlace-v0", embodiment="galaxea_r1lite"),
    "r1pro": dict(env="R1ProCubeBowlPlace-v0", embodiment="galaxea_r1pro"),
}
VIEW_HEIGHT = 360
LABEL_FONT = ImageFont.load_default(size=18)


class SimVLARunner:
    """Owns the env and the policy connection. All sim calls happen on one thread."""

    def __init__(self, robot: str, server: str, max_steps: int):
        self.cfg = ROBOTS[robot]
        self.max_steps = max_steps
        self.env = gym.make(self.cfg["env"], headless=True, obs_mode="state")
        self.u = self.env.unwrapped
        self.policy = G05Policy(server, self.cfg["embodiment"])
        self.lock = threading.Lock()
        self.frame = None  # latest JPEG bytes
        self.frame_cond = threading.Condition()
        self.status = dict(state="idle", prompt="", step=0, success=False, message="")
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
        self._reset_scene(new_layout=True)
        self._set(message="new scene")
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
                        self._reset_scene(new_layout=True)
                        self._set(state="idle", step=0, success=False, message="new scene")
                    elif kind == "run":
                        self._run(prompt)
                except Exception as e:  # keep the UI alive and report the error
                    self._set(state="error", message=f"{type(e).__name__}: {e}")
                    self.policy.ws = None

    def _set(self, **kw):
        with self.lock:
            self.status.update(kw)

    def _reset_scene(self, new_layout):
        """Random new cube/bowl layout, or the current layout again. The robot starts at rest."""
        if new_layout:
            _, self.reset_info = self.env.reset()
        else:
            self.u.robot.set_qpos(self.u.init_qpos)
            self.u.robot.robot.set_qvel(np.zeros(len(self.u.init_qpos)))
            self.u.controller.reset()
            self.u._init_buffers()
            self.u.reset_world(self.reset_info)
            self.u._scene.update_render()
        self._publish_frame()

    def _run(self, prompt):
        self._reset_scene(new_layout=False)
        self.policy.reset(self.u)
        self._set(state="running", prompt=prompt, step=0, success=False, message="")
        for step in range(1, self.max_steps + 1):
            if self.stop_event.is_set():
                self._set(state="stopped", message=f"stopped at step {step - 1}")
                return
            info = self.u.step(self.policy.act(self.u, prompt)[0])[-1]
            self._set(step=step, success=bool(info["success"]))
            self._publish_frame()
            if info["success"]:
                self._set(state="done", message=f"success at step {step}")
                return
        self._set(state="done", message=f"no success after {self.max_steps} steps")

    def _publish_frame(self):
        """2x2 grid: scene | head (top), left wrist | right wrist (bottom), each row centered."""
        head, left, right = self.u.cameras
        rows = [
            [("scene", self.u.default_camera), ("head (G0.5)", head)],
            [("left wrist (G0.5)", left), ("right wrist (G0.5)", right)],
        ]
        tiles = [
            [(label, Image.fromarray(img).resize((int(img.shape[1] * VIEW_HEIGHT / img.shape[0]), VIEW_HEIGHT)))
             for label, img in ((label, rgb(cam)) for label, cam in row)]
            for row in rows
        ]
        width = max(sum(t.width for _, t in row) for row in tiles)
        canvas = Image.new("RGB", (width, VIEW_HEIGHT * len(tiles)), (17, 17, 17))
        draw = ImageDraw.Draw(canvas)
        for r, row in enumerate(tiles):
            x = (width - sum(t.width for _, t in row)) // 2
            for label, tile in row:
                canvas.paste(tile, (x, r * VIEW_HEIGHT))
                box = draw.textbbox((x + 8, r * VIEW_HEIGHT + 6), label, font=LABEL_FONT)
                draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
                draw.text((x + 8, r * VIEW_HEIGHT + 6), label, font=LABEL_FONT, fill=(255, 255, 255))
                x += tile.width
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
</style></head><body>
<h3>__TITLE__</h3>
<img src="/stream" alt="sim views: scene, head, left wrist, right wrist">
<form onsubmit="post('/run');return false">
<input id="p" value="__PROMPT__"><button>Run</button>
<button type="button" onclick="post('/stop')">Stop</button>
<button type="button" onclick="post('/reset')">New scene</button></form>
<div id="s"></div>
<script>
function post(u){fetch(u,{method:'POST',body:document.getElementById('p').value})}
setInterval(async()=>{const s=await (await fetch('/status')).json();
document.getElementById('s').textContent=`${s.state} | step ${s.step} | success ${s.success} | ${s.message}`;},500);
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
    ap.add_argument("--max-steps", type=int, default=1200, help="steps per prompt (15 Hz)")
    args = ap.parse_args()

    runner = SimVLARunner(args.robot, args.server, args.max_steps)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner, f"{runner.cfg['env']} ← G0.5"))
    httpd.daemon_threads = True  # open /stream connections must not keep the process alive after Ctrl+C
    print(f"Open http://127.0.0.1:{args.port} (policy server: {args.server})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
