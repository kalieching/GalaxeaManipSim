"""Play back a recorded kitting trajectory: restore every step and render a video of the env's external camera (in
front of the robot, off to its right, whole body in view) above the head and wrist cameras, captioned with the step,
the prompt in effect and how many objects are packed.

Works for demos (datasets/<Robot>Kitting-v0/<tag>/demo_<n>.h5) and failed attempts (failed/<name>.h5). The robot
comes from the folder name and reset_info from the matching meta_info.json. --start plays back from a later step, and
the same restore() call resumes a trajectory from that step (e.g. to run a policy onward).

--cameras also writes every step of each robot camera, at full resolution and near-lossless (H.264, CRF 15), for
training: <stem>_cameras/{head_rgb,left_wrist_rgb,right_wrist_rgb}.mp4, frame t = step t of the h5 (15 fps).

Usage: python xmem/replay_kitting.py datasets/R1LiteKitting-v0/g05/demo_0.h5 [--start 0] [--every 1] [--cameras]
       [--out demo_0.mp4]
"""
import argparse
import contextlib
import json
from pathlib import Path

import gymnasium as gym
import h5py
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import galaxea_sim.envs  # noqa: F401  (registers envs)
from policy_client import rgb

FONT = ImageFont.load_default(size=26)
CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")  # env.cameras order, G0.5's names


def load_step(h5, t):
    """object_dict, sim_state and upper_body_action_dict of step t, as restore() takes them."""
    def read(group):
        return {k: read(v) if isinstance(v, h5py.Group) else v[t] for k, v in group.items()}
    return read(h5["object_dict"]), read(h5["sim_state"]), read(h5["upper_body_action_dict"])


def reset_info_for(path):
    meta = json.loads((path.parent / "meta_info.json").read_text())
    if path.parent.name == "failed":
        return next(m for m in meta if m["file"] == path.name)["reset_info"]
    return meta[int(path.stem.split("_")[-1])]["reset_info"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5", type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="render every n-th step")
    ap.add_argument("--cameras", action="store_true", help="also write each robot camera at every step, for training")
    ap.add_argument("--out", type=Path, help="video path (default: next to the h5)")
    args = ap.parse_args()
    assert not args.cameras or (args.start, args.every) == (0, 1), "--cameras keeps every step from step 0"

    env_id = next(p for p in args.h5.resolve().parts if p.endswith("Kitting-v0"))
    env = gym.make(env_id, headless=True, obs_mode="state")
    u = env.unwrapped
    reset_info = reset_info_for(args.h5)
    out = args.out or args.h5.with_suffix(".mp4")
    with contextlib.ExitStack() as stack:
        h5 = stack.enter_context(h5py.File(args.h5, "r"))
        video = stack.enter_context(imageio.get_writer(out, fps=15 / args.every))
        streams = []
        if args.cameras:
            folder = args.h5.with_name(args.h5.stem + "_cameras")
            folder.mkdir(exist_ok=True)
            streams = [stack.enter_context(imageio.get_writer(
                folder / f"{name}.mp4", fps=15, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                ffmpeg_params=["-crf", "15", "-preset", "slow"])) for name in CAMERAS]
        n = len(h5["sim_state"]["robot_qpos"])
        for t in range(args.start, n, args.every):
            u.restore(reset_info, *load_step(h5, t))
            head, left, right = images = [rgb(c) for c in u.cameras]
            for stream, image in zip(streams, images):
                stream.append_data(image)
            wrists = np.concatenate([left, right], axis=1)
            scale = head.shape[1] // wrists.shape[1]
            scene = Image.fromarray(rgb(u.default_camera))
            scene = np.asarray(scene.resize((head.shape[1], scene.height * head.shape[1] // scene.width)))
            frame = Image.fromarray(np.concatenate([np.zeros((48, head.shape[1], 3), np.uint8), scene, head,
                                                    wrists.repeat(scale, 0).repeat(scale, 1)], axis=0))
            packed = sum(u._get_info()["inside"])
            caption = f"step {t}   {h5['language_instruction'][t].decode()}   packed {packed}/{len(u.objects)}"
            ImageDraw.Draw(frame).text((12, 10), caption, font=FONT, fill=(255, 255, 255))
            video.append_data(np.asarray(frame))
    print(f"wrote {out} (steps {args.start}..{n - 1})")


if __name__ == "__main__":
    main()
