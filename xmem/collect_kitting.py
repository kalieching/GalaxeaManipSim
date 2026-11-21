"""Collect kitting trajectories with G0.5 or a scripted planner: one per planned scene, all objects packed, any order.

The R1 Lite and R1 Pro get the same scenes and layouts (kitting_layouts.json) and are collected the same way.
--policy g05 needs a policy server (bash xmem/up.sh --robot r1lite --server-only). --policy scripted plans each
pick-and-place with mplib (xmem/scripted_kitting.py). Each attempt targets one object that is not packed yet,
preferring the object that has failed least. An attempt succeeds when that object is in the container
and released and every object packed before is still in. Otherwise, after --max-steps (or when the script ends or cannot plan), the simulation goes back to
where the attempt began (KittingEnv.restore) and the next attempt starts. A kit trajectory is therefore the successful
attempts, each starting exactly where the previous one ended. After --max-attempts the layout is dropped and the scene
starts over with a new one, up to --layouts times. Scenes already in meta_info.json are skipped, so an interrupted run
can be continued.

Writes to --out (default datasets/<Robot>Kitting-v0/<policy>), in upstream's layout (replay_demos and the LeRobot converters read it):
  demo_<n>.h5             every step: the env observation (joint states, commands, end-effector, object and camera
                          poses, language instruction, sim_state), the target object index, and G0.5's raw action (NaN
                          where G0.5 left a part out, all NaN for the scripted policy)
  g05_inputs/demo_<n>.h5  (G0.5 only) every G0.5 query: the step it chose actions for, the prompt, and the images and joint
                          state sent (a subfolder, so replay_demos does not take it for a demo)
  meta_info.json          per demo: scene, layout, reset_info (rebuilds the layout, camera intrinsics), prompts, and
                          each placement (object, first and last step, attempts it took)
  failed/                 every failed attempt, the same per-step data (its row 0 is the state it began from),
                          g05_inputs/ alongside, and failed/meta_info.json: scene, layout, the demo it belongs to (null
                          if the layout was dropped), object, attempt number, and the demo step it branched off at
Play back or resume at step t: env.unwrapped.restore(reset_info, object_dict[t], sim_state[t],
upper_body_action_dict[t]) (see xmem/replay_kitting.py).

Usage: python xmem/collect_kitting.py --robot r1lite|r1pro [--policy g05|scripted] [--scenes 0 1 ...] [--out DIR]
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

import galaxea_sim.envs  # noqa: F401  (registers envs)
from galaxea_sim.envs.robotwin.kitting import LAYOUTS
from galaxea_sim.utils.data_utils import save_dict_list_to_hdf5, save_dict_list_to_json
from policy_client import G05Policy
from scripted_kitting import ScriptedPolicy

ROBOTS = dict(r1lite=("R1LiteKitting-v0", "galaxea_r1lite"), r1pro=("R1ProKitting-v0", "galaxea_r1pro"))


def raw_action(action, arm_dof):
    sizes = dict(left_arm=arm_dof, left_gripper=1, right_arm=arm_dof, right_gripper=1)  # R1 Lite arms 6, R1 Pro 7
    return {k: np.asarray(action[k], np.float32).reshape(-1) if k in action else np.full(n, np.nan, np.float32)
            for k, n in sizes.items()}


class Collector:
    def __init__(self, args):
        self.args = args
        env_id, embodiment = ROBOTS[args.robot]
        args.out = args.out or Path("datasets") / env_id / args.policy
        self.env = gym.make(env_id, headless=True, obs_mode="state")
        self.u = self.env.unwrapped
        self.policy = G05Policy(args.server, embodiment) if args.policy == "g05" else ScriptedPolicy(args.seed)
        self.meta_path, self.failed_path = args.out / "meta_info.json", args.out / "failed" / "meta_info.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else []
        self.failed = json.loads(self.failed_path.read_text()) if self.failed_path.exists() else []

    def attempt(self, target, steps, queries):
        """Run one attempt from the current state, appending to steps/queries. True if it packed the target."""
        u, prompt = self.u, self.u.prompts[target]
        u.target = target
        before = u._get_info()["inside"]
        self.policy.reset(u)
        for _ in range(self.args.max_steps):
            action, sent, g05_action = self.policy.act(u, prompt)
            if action is None:  # the script finished or could not plan
                break
            if sent is not None:
                queries.append(dict(step=len(steps), prompt=prompt, state=sent["state"],
                                    images={k: v.transpose(1, 2, 0) for k, v in sent["images"].items()}))
            obs, _, _, _, info = self.env.step(action)
            obs.update(target=target, g05_action=raw_action(g05_action, len(u.left_arm_joint_indices)))
            steps.append(obs)
            inside = info["inside"]
            if inside[target] and all(i for i, b in zip(inside, before) if b):
                return True
        return False

    def save_failure(self, scene, layout, target, attempt, branch_step, start_obs, steps, queries, reset_info):
        name = f"scene{scene}_layout{layout}_object{target}_attempt{attempt}"
        # row 0: the state the attempt began from
        first = dict(start_obs, target=target, g05_action=raw_action({}, len(self.u.left_arm_joint_indices)))
        save_dict_list_to_hdf5([first] + steps, self.args.out / "failed" / f"{name}.h5")
        if queries:
            for q in queries:
                q["step"] += 1
            save_dict_list_to_hdf5(queries, self.args.out / "failed" / "g05_inputs" / f"{name}.h5")
        self.failed.append(dict(file=f"{name}.h5", scene=scene, layout=layout, demo=None, object=target,
                                object_name=self.u.objects[target]["name"], attempt=attempt, branch_step=branch_step,
                                steps=len(steps), reset_info=reset_info))
        save_dict_list_to_json(self.failed, self.failed_path)

    def run_scene(self, scene):
        """Pack the scene's objects. Returns (steps, queries, meta) or None if no layout worked."""
        u, args = self.u, self.args
        assert args.layouts <= len(LAYOUTS.get(str(scene), [])), "more layouts than kitting_layouts.json has: not paired"
        for layout in range(args.layouts):
            obs, reset_info = self.env.reset(seed=layout, options=dict(scene=scene))
            steps, queries, placements = [], [], []
            fails = [0] * len(u.objects)
            for n_attempt in range(args.max_attempts):
                inside = u._get_info()["inside"]
                if all(inside):
                    break
                target = min((i for i, x in enumerate(inside) if not x), key=lambda i: fails[i])
                start = (len(steps), len(queries), obs)
                ok = self.attempt(target, steps, queries)
                print(f"  scene {scene} layout {layout} attempt {n_attempt}: {u.objects[target]['name']} "
                      f"{'packed' if ok else 'not packed'} (kit at step {len(steps) if ok else start[0]})", flush=True)
                if ok:
                    placements.append(dict(object=target, name=u.objects[target]["name"], first_step=start[0],
                                           last_step=len(steps) - 1, attempts=fails[target] + 1))
                    obs = steps[-1]
                    continue
                self.save_failure(scene, layout, target, fails[target], start[0], start[2], steps[start[0]:],
                                  queries[start[1]:], reset_info)
                fails[target] += 1
                del steps[start[0]:], queries[start[1]:]
                obs = u.restore(reset_info, start[2]["object_dict"], start[2]["sim_state"],
                                start[2]["upper_body_action_dict"])
            if all(u._get_info()["inside"]):
                return steps, queries, dict(reset_info=reset_info, success=True, total_steps=len(steps), scene=scene,
                                            layout=layout, prompts=reset_info["prompts"], placements=placements,
                                            failed_attempts=sum(fails), layouts_tried=layout + 1)
        return None

    def run(self):
        done = {m["scene"] for m in self.meta}
        scenes = self.args.scenes if self.args.scenes is not None else range(len(self.u.plan))
        for scene in scenes:
            if scene in done:
                continue
            t0 = time.time()
            result = self.run_scene(scene)
            if result is None:
                print(f"scene {scene}: no full kit in {self.args.layouts} layouts", flush=True)
                continue
            steps, queries, info = result
            n = len(self.meta)
            save_dict_list_to_hdf5(steps, self.args.out / f"demo_{n}.h5")
            if queries:
                save_dict_list_to_hdf5(queries, self.args.out / "g05_inputs" / f"demo_{n}.h5")
            self.meta.append(info)
            save_dict_list_to_json(self.meta, self.meta_path)
            for f in self.failed:
                if f["scene"] == scene and f["layout"] == info["layout"]:
                    f["demo"] = n
            save_dict_list_to_json(self.failed, self.failed_path)
            print(f"scene {scene}: saved demo_{n} ({len(steps)} steps, {info['failed_attempts']} failed attempts, "
                  f"{time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--robot", choices=ROBOTS, required=True)
    ap.add_argument("--out", type=Path, help="default: datasets/<Robot>Kitting-v0/g05")
    ap.add_argument("--policy", choices=("g05", "scripted"), default="g05")
    ap.add_argument("--server", default="ws://127.0.0.1:8765", help="G0.5 policy server")
    ap.add_argument("--seed", type=int, default=0, help="scripted policy random choices")
    ap.add_argument("--scenes", type=int, nargs="*", help="scene ids (default: all planned)")
    ap.add_argument("--max-steps", type=int, default=600, help="steps per attempt (15 Hz)")
    ap.add_argument("--max-attempts", type=int, default=20, help="attempts per layout before trying a new layout")
    ap.add_argument("--layouts", type=int, default=3, help="layouts to try per scene")
    Collector(ap.parse_args()).run()


if __name__ == "__main__":
    main()
