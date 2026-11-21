"""Kitting: put three objects into a container, one prompt per object.

A scene is one container in the middle of the table with three objects beside it, two on one side and one on the
other, in front of the arm on that side, like CubeBowlPlace's cube and bowl, where G0.5 succeeds. Scenes alternate
between two boxy and one round object and one boxy and two round. plan_scenes() fixes which container and objects each
scene uses, mixing 4 containers and 12 objects evenly. reset(options=
dict(scene=i), seed=s) loads layout s of scene i from kitting_layouts.json and settles it on the robot's table.
Those layouts are shared by the R1 Lite and R1 Pro (positions relative to the robot base), and each passes, for both
robots: the container's footprint and every object, base and top, are in the head camera image, an arm can reach
every object's mid-height and a drop point above the container (fingers straight down or tilted 30/45 deg), and
nothing overlaps. python -m galaxea_sim.envs.robotwin.kitting [scene ...] regenerates the file (or just
those scenes).

Objects can be packed in any order. `target` (an object index) picks the object the language instruction asks for.
By default it is the first object not yet in the container.

reset_info rebuilds a layout exactly (reset(options=dict(reset_info=...))) and holds each camera's intrinsics. Every
observation carries sim_state and every camera's world pose, and restore() puts the whole simulation back to any
recorded step, so a trajectory can be played back or resumed anywhere.
Meshes come from RoboTwin 2.0 (python -m galaxea_sim.scripts.fetch_assets).
"""
import json
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
from mplib.pymp import Pose

from galaxea_sim import ASSETS_DIR
from galaxea_sim.planners.bimanual import BimanualPlanner
from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.utils.sapien_utils import get_link_by_name
from .cube_bowl_place import CubeBowlPlaceEnv
from .robotwin_base import RoboTwinBaseEnv

MODELS_DIR = ASSETS_DIR / "robotwin2_models"
# (RoboTwin 2.0 model, variant, name used in prompts). "cube" is a plain 4 cm red box
CONTAINERS = [("063_tabletrashbin", 0, "orange bin"), ("007_shoe-box", 0, "orange shoe box"),
              ("110_basket", 1, "yellow basket"), ("062_plasticbox", 2, "white box")]
BOXY = [("cube", 0, "red cube"), ("073_rubikscube", 0, "Rubik's cube"), ("086_woodenblock", 1, "wooden block"),
        ("112_tea-box", 1, "red tea box"), ("112_tea-box", 2, "teal tea box"), ("113_coffee-box", 6, "white coffee box")]
ROUND = [("071_can", 6, "red can"), ("071_can", 0, "white can"), ("071_can", 5, "brown can"),
         ("105_sauce-can", 5, "sauce jar"), ("059_pencup", 5, "blue pen cup"),
         ("109_hydrating-oil", 0, "yellow bottle")]
KITTING_MODELS = [(m, v) for m, v, _ in CONTAINERS + BOXY + ROUND if m != "cube"]
BASKET_HANDLE = ("110_basket", 1, 0.14)  # fetch_assets cuts everything above 14 cm off this basket
PROMPT = "pick up the {object} and place it in the {container}"

N_OBJECTS = 3  # per scene: even scenes get 2 boxy + 1 round, odd scenes 1 boxy + 2 round
# Planned object replaced in a scene: G0.5 on the R1 Lite never packed the Rubik's cube into scene 4's white box
# (52 attempts over 3 layouts), so the red cube takes its place. Each boxy object stays in 2 or 3 scenes.
SWAPS = {4: ("Rubik's cube", BOXY[0])}
# The round slot that held the pink perfume bottle (115_perfume 0) now holds the blue pen cup: the R1 Pro could not
# grip the bottle (only its 4.6 cm side fits, and it stands 15.5 cm tall). It was in scenes 1 and 5.
CUBE_HALF = 0.02
CONTAINER_Y = 0.04          # |y| of the container center, m
CONTAINER_X = (0.47, 0.60)  # container center, m in front of the robot base (CubeBowlPlace's bowl: 0.48-0.53)
OBJECT_X = (0.44, 0.60)     # object centers, m in front of the robot base (CubeBowlPlace's cube: 0.45-0.54)
OBJECT_Y = 0.40             # max |y| of an object's footprint edge, m (room for two objects beside the container)
GAP = 0.02                  # between any two footprints, m
DROP_ABOVE_RIM = 0.08       # m
IMG_MARGIN = 30             # px kept clear at the head camera image border
TILTS = (0, 30, 45)         # gripper tilt from straight down, deg
INSIDE_MARGIN = 0.01        # an object's center must be this far inside the container's walls, m
SETTLE_S = 1.0
Y_UP_TO_Z_UP = [0.7071068, 0.7071068, 0, 0]  # RoboTwin meshes are y-up
# Layouts shared by every robot (python -m galaxea_sim.envs.robotwin.kitting writes them)
LAYOUTS_PATH = Path(__file__).with_name("kitting_layouts.json")
LAYOUTS = json.loads(LAYOUTS_PATH.read_text())["layouts"] if LAYOUTS_PATH.exists() else {}


def plan_scenes(n_scenes=10, seed=0):
    """Container and objects of each scene: containers and objects each used as evenly as possible."""
    rng = np.random.default_rng(seed)

    def draws(pool, ks):
        counts, out = np.zeros(len(pool), int), []
        for k in ks:
            pick = sorted(range(len(pool)), key=lambda i: (counts[i], rng.random()))[:k]
            counts[pick] += 1
            out.append([pool[i] for i in rng.permutation(pick)])
        return out

    containers = [CONTAINERS[i % len(CONTAINERS)] for i in rng.permutation(n_scenes)]
    n_boxy = [2 - i % 2 for i in range(n_scenes)]
    boxy, rnd = draws(BOXY, n_boxy), draws(ROUND, [N_OBJECTS - k for k in n_boxy])
    plan = [dict(container=c, objects=b + r) for c, b, r in zip(containers, boxy, rnd)]
    for scene, (old, new) in SWAPS.items():
        if scene < n_scenes:
            plan[scene]["objects"] = [new if o[2] == old else o for o in plan[scene]["objects"]]
    return plan


def model_box(model, variant):
    """Model data, and the size (x, y, z) and mesh-frame center of its box once stood upright."""
    if model == "cube":
        return None, np.full(3, 2 * CUBE_HALF), np.zeros(3)
    data = json.loads((MODELS_DIR / model / f"model_data{variant}.json").read_text())
    scale = np.asarray(data["scale"], float)
    extents = np.asarray(data["extents"], float) * scale
    return data, extents[[0, 2, 1]], np.asarray(data["center"], float) * scale


def sample_layout(plan, rng, envs):
    """Positions (x, y in the robot base frame, yaw) for the planned container and objects that pass the camera and
    reach checks for every env in envs, each on its own table. None if an item finds no spot."""
    layout, taken = [], []
    model, variant, name = plan["container"]
    _, size, _ = model_box(model, variant)
    hx, hy, h = size / 2 * [1, 1, 2]
    for _ in range(300):
        cx, cy = rng.uniform(*CONTAINER_X), rng.uniform(-CONTAINER_Y, CONTAINER_Y)
        if all(e._box_visible(cx, cy, hx, hy, h) and e._reachable([cx, cy, e.table_top_z + h + DROP_ABOVE_RIM])
               for e in envs):
            break
    else:
        return None
    layout.append(dict(kind="container", model=model, variant=variant, name=name, x=cx, y=cy, yaw=0.0))
    taken.append((cx, cy, hx, hy))
    # objects on both sides of the container (+1: the robot's left), at least one per side
    sides = rng.permutation([1, -1] + list(rng.choice([1, -1], len(plan["objects"]) - 2)))
    for (model, variant, name), side in zip(plan["objects"], sides):
        _, size, _ = model_box(model, variant)
        for _ in range(400):
            yaw = rng.uniform(0, np.pi)
            c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
            ex, ey = c * size[0] / 2 + s * size[1] / 2, s * size[0] / 2 + c * size[1] / 2
            near = side * cy + hy + GAP + ey  # |y| just clear of the container on this side
            if near > OBJECT_Y - ey:
                continue
            x, y = rng.uniform(*OBJECT_X), side * rng.uniform(near, OBJECT_Y - ey)
            if any(abs(x - tx) < ex + thx + GAP and abs(y - ty) < ey + thy + GAP for tx, ty, thx, thy in taken):
                continue
            if all(e._box_visible(x, y, ex, ey, size[2]) and e._reachable([x, y, e.table_top_z + size[2] / 2])
                   for e in envs):
                break
        else:
            return None
        layout.append(dict(kind="object", model=model, variant=variant, name=name, x=x, y=y, yaw=yaw))
        taken.append((x, y, ex, ey))
    return layout


def generate_layouts(scenes=None, n_layouts=5, robots=("R1Lite", "R1Pro")):
    """Sample n_layouts layouts per planned scene that suit every robot and write them to kitting_layouts.json.
    scenes: only (re)generate these, keeping the file's other scenes."""
    import gymnasium as gym
    envs = [gym.make(f"{r}Kitting-v0", headless=True, obs_mode="state").unwrapped for r in robots]
    for e in envs:
        e._reset_robot()
        e._scene.update_render()
    layouts = dict(LAYOUTS) if scenes else {}
    for scene, plan in enumerate(envs[0].plan):
        if scenes and scene not in scenes:
            continue
        rng = np.random.default_rng([scene, 1000])
        layouts[str(scene)] = []
        while len(layouts[str(scene)]) < n_layouts:
            layout = sample_layout(plan, rng, envs)
            if layout is not None:
                layouts[str(scene)].append(layout)
        print(f"scene {scene}: {n_layouts} layouts for {', '.join(robots)}", flush=True)
    LAYOUTS_PATH.write_text(json.dumps(dict(robots=list(robots), layouts=layouts), indent=1))


class KittingEnv(RoboTwinBaseEnv):
    def __init__(self, *args, n_scenes=10, plan_seed=0, **kwargs):
        self.plan = plan_scenes(n_scenes, plan_seed)
        self.items, self.scene_id, self.layout_seed, self._planner, self.target = [], 0, 0, None, None
        self._contacts_valid = False  # SAPIEN's contact list points at removed actors until the next physics step
        super().__init__(*args, **kwargs)
        # same stiff arm drives as CubeBowlPlace, tuned so G0.5's 15 Hz targets are reached like on the real robot
        for i in list(self.left_arm_joint_indices) + list(self.right_arm_joint_indices):
            self.active_joints[i].set_drive_property(stiffness=CubeBowlPlaceEnv.arm_joint_stiffness,
                                                     damping=CubeBowlPlaceEnv.arm_joint_damping)

    # ---- scene ----
    def reset(self, *, seed=None, options=None):
        """options: scene=i (with seed choosing the layout), or reset_info=... to rebuild a recorded layout."""
        options = options or {}
        self._reset_robot()
        if "reset_info" in options:
            self.reset_world(options["reset_info"])
        else:
            self.scene_id, self.layout_seed = options.get("scene", self.scene_id), seed or 0
            self.reset_world()
        self._scene.update_render()
        return self._get_obs(), self._get_reset_info()

    def _reset_robot(self):
        self.robot.set_qpos(self.init_qpos)
        self.robot.robot.set_qvel(np.zeros(self.num_dofs))
        self.controller.reset()
        self._init_buffers()
        for joint, target in zip(self.active_joints, self.init_qpos):
            joint.set_drive_target(target)

    def step(self, action):
        self._contacts_valid = True
        return super().step(action)

    def reset_world(self, reset_info=None):
        self._contacts_valid = False
        for it in self.items:
            self._scene.remove_actor(it["actor"])
        self.items = []
        if not hasattr(self, "robot"):  # the first reset_world runs before the robot exists
            return
        if reset_info is not None:
            self.scene_id, self.layout_seed = int(reset_info["scene"]), int(reset_info["layout_seed"])
            for spec in reset_info["items"]:
                it = self._add_item(spec["kind"], spec["model"], int(spec["variant"]), spec["name"])
                it["actor"].set_pose(sapien.Pose(spec["pose"][:3], spec["pose"][3:]))
            return
        layouts = LAYOUTS.get(str(self.scene_id), [])
        if self.layout_seed < len(layouts):  # paired: the same layout for every robot
            layout = layouts[self.layout_seed]
        else:  # not in kitting_layouts.json: sample one for this robot only
            rng = np.random.default_rng([self.scene_id, self.layout_seed])
            layout = next(filter(None, (sample_layout(self.plan[self.scene_id], rng, [self]) for _ in range(50))), None)
            if layout is None:
                raise RuntimeError(f"no valid layout for scene {self.scene_id} (layout seed {self.layout_seed})")
        for spec in layout:
            self._place(self._add_item(spec["kind"], spec["model"], spec["variant"], spec["name"]),
                        spec["x"], spec["y"], spec["yaw"])
        for _ in range(int(SETTLE_S / self.timestep)):
            self.robot.set_qf(self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self._scene.step()

    def _add_item(self, kind, model, variant, name):
        key = "container" if kind == "container" else f"object_{len([i for i in self.items if i['kind'] == 'object'])}"
        if model == "cube":
            actor = create_box(self._scene, sapien.Pose(), half_size=(CUBE_HALF,) * 3, color=(1, 0, 0), name=key)
            _, size, center = model_box(model, variant)
        else:
            data, size, center = model_box(model, variant)
            builder = self._scene.create_actor_builder()
            builder.set_physx_body_type("static" if kind == "container" else "dynamic")
            builder.add_multiple_convex_collisions_from_file(
                str(MODELS_DIR / model / "collision" / f"base{variant}.glb"), scale=data["scale"], decomposition="none")
            builder.add_visual_from_file(str(MODELS_DIR / model / "visual" / f"base{variant}.glb"), scale=data["scale"])
            actor = builder.build(name=key)
        item = dict(key=key, kind=kind, model=model, variant=variant, name=name, actor=actor, size=size,
                    center=center, mesh_turn=sapien.Pose() if model == "cube" else sapien.Pose(q=Y_UP_TO_Z_UP))
        self.items.append(item)
        return item

    @property
    def table_top_z(self):
        return self.tabletop_center_in_world[2]

    def _place(self, item, x, y, yaw=0.0):
        """Stand the item on the table with its box centered at (x, y), turned by yaw."""
        q_yaw = sapien.Pose(q=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        z = self.tabletop_center_in_world[2] + item["size"][2] / 2 + 0.002
        item["actor"].set_pose(sapien.Pose([x, y, z]) * q_yaw * item["mesh_turn"] * sapien.Pose(-item["center"]))

    def _box_visible(self, x, y, hx, hy, h):
        cam = self.cameras[0]
        z0 = self.tabletop_center_in_world[2]
        pts = np.array([[x + sx * hx, y + sy * hy, z0 + z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, h)])
        ext = cam.get_extrinsic_matrix()
        pc = pts @ ext[:3, :3].T + ext[:3, 3]
        if (pc[:, 2] <= 0).any():
            return False
        uv = pc @ cam.get_intrinsic_matrix().T
        uv = uv[:, :2] / uv[:, 2:3]
        return bool(((uv > IMG_MARGIN) & (uv < [cam.width - IMG_MARGIN, cam.height - IMG_MARGIN])).all())

    def _grasp_quats(self, side, tilts=TILTS):
        """End-effector orientations that point this arm's fingers straight down or tilted `tilts` deg forward, with the
        fingers closing sideways (along world y). Found from the finger links, so they work for any robot whose
        end-effector frame is oriented differently (R1 Lite's x axis points along the fingers, R1 Pro's does not).
        Uses the robot's current pose, which is the initial pose while a layout is sampled."""
        ee = getattr(self.robot, f"{side}_ee_link").entity_pose
        f1, f2 = (get_link_by_name(self.robot.links, f"{side}_gripper_finger_link{i}").entity_pose.p for i in (1, 2))
        to_local = t3d.quaternions.quat2mat(ee.q).T
        a = to_local @ ((f1 + f2) / 2 - ee.p)  # fingers' direction, in the end-effector frame
        c = to_local @ (f1 - f2)               # closing direction
        a /= np.linalg.norm(a)
        c -= (c @ a) * a
        c /= np.linalg.norm(c)
        local = np.stack([a, c, np.cross(a, c)], axis=1)
        y = np.sign(t3d.quaternions.quat2mat(ee.q) @ c @ [0, 1, 0]) or 1.0  # keep the finger order it starts with
        quats = []
        for tilt in np.deg2rad(tilts):
            d = np.array([np.sin(tilt), 0, -np.cos(tilt)])
            side_axis = np.array([0, y, 0])
            world = np.stack([d, side_axis, np.cross(d, side_axis)], axis=1)
            quats.append(t3d.quaternions.mat2quat(world @ local.T))
        return quats

    def _reachable(self, p):
        if self._planner is None:
            self._planner = BimanualPlanner(
                urdf_path=f"{self.robot.name}/robot.urdf", srdf_path=None, left_arm_move_group=self.left_ee_link_name,
                right_arm_move_group=self.right_ee_link_name, active_joint_names=self.active_joint_names,
                control_freq=self.control_freq)
            self._grasps = {side: self._grasp_quats(side) for side in ("left", "right")}
        pl = self._planner
        qpos = pl.sim2mplib_mapping(np.asarray(self.init_qpos, float))
        arms = (("left", pl.left_arm_planner, pl.left_arm_planner_mask),
                ("right", pl.right_arm_planner, pl.right_arm_planner_mask))
        return any(planner.IK(Pose(p=p, q=q), qpos, mask=mask, n_init_qpos=6)[0] == "Success"
                   for side, planner, mask in arms for q in self._grasps[side])

    # ---- state ----
    def _box_frame(self, item):
        """Pose of the item's upright box: centered, z up (undoes the mesh's y-up turn)."""
        return item["actor"].get_pose() * sapien.Pose(item["center"]) * item["mesh_turn"].inv()

    def _get_reset_info(self):
        cameras = {c.name: dict(width=c.width, height=c.height, intrinsic=c.get_intrinsic_matrix()) for c in self.cameras}
        return dict(scene=self.scene_id, layout_seed=self.layout_seed, prompts=self.prompts, cameras=cameras,
                    items=[dict(kind=it["kind"], model=it["model"], variant=it["variant"], name=it["name"],
                                pose=np.concatenate([it["actor"].get_pose().p, it["actor"].get_pose().q]))
                           for it in self.items])

    def get_object_dict(self):
        return {it["key"]: np.concatenate([it["actor"].get_pose().p, it["actor"].get_pose().q]) for it in self.items}

    def _get_obs(self):
        obs = super()._get_obs()
        obs["sim_state"] = self.get_sim_state()
        # world pose (x, y, z, qw, qx, qy, qz) of each camera, x looking forward, y left, z up (SAPIEN convention)
        obs["camera_pose"] = {c.name: np.concatenate([c.get_global_pose().p, c.get_global_pose().q]) for c in self.cameras}
        return obs

    def get_sim_state(self):
        """Everything restore() needs besides the commands in upper_body_action_dict."""
        velocities = {}
        for it in self.items:
            body = it["actor"].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if body is not None:
                velocities[it["key"]] = np.concatenate([body.linear_velocity, body.angular_velocity])
        return dict(robot_qpos=self.robot.get_qpos(), robot_qvel=self.robot.robot.get_qvel(),
                    object_velocity=velocities)

    def restore(self, reset_info, object_dict, sim_state, action_dict):
        """Put the simulation back to a recorded step: layout from reset_info, then that step's object poses, robot
        joints, velocities and commands (an observation's object_dict, sim_state and upper_body_action_dict)."""
        self.reset(options=dict(reset_info=reset_info))
        for it in self.items:
            pose = np.asarray(object_dict[it["key"]])
            it["actor"].set_pose(sapien.Pose(pose[:3], pose[3:]))
            body = it["actor"].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if body is not None and it["key"] in sim_state["object_velocity"]:
                v = np.asarray(sim_state["object_velocity"][it["key"]])
                body.set_linear_velocity(v[:3])
                body.set_angular_velocity(v[3:])
        self.robot.set_qpos(np.asarray(sim_state["robot_qpos"]))
        self.robot.robot.set_qvel(np.asarray(sim_state["robot_qvel"]))
        cmd = {k: np.asarray(v).reshape(-1) for k, v in action_dict.items()}
        targets = list(self.init_qpos)
        for side in ("left", "right"):
            for i, j in enumerate(getattr(self, f"{side}_arm_joint_indices")):
                targets[j] = cmd[f"{side}_arm_joint_position_cmd"][i]
            grip = float(cmd[f"{side}_arm_gripper_position_cmd"][0])
            for j, sign in zip(getattr(self, f"{side}_gripper_joint_indices"), self.robot.gripper_finger_sign):
                targets[j] = grip * sign
        for joint, target in zip(self.active_joints, targets):
            joint.set_drive_target(target)
        self.left_arm_joint_position_cmd = cmd["left_arm_joint_position_cmd"]
        self.right_arm_joint_position_cmd = cmd["right_arm_joint_position_cmd"]
        self.left_arm_gripper_position_cmd = float(cmd["left_arm_gripper_position_cmd"][0])
        self.right_arm_gripper_position_cmd = float(cmd["right_arm_gripper_position_cmd"][0])
        self.last_gripper_cmd = [self.left_arm_gripper_position_cmd, self.right_arm_gripper_position_cmd]
        self._scene.update_render()
        return self._get_obs()

    # ---- task ----
    @property
    def objects(self):
        return [it for it in self.items if it["kind"] == "object"]

    @property
    def prompts(self):
        container = [it for it in self.items if it["kind"] == "container"]
        if not container:
            return []
        return [PROMPT.format(object=o["name"], container=container[0]["name"]) for o in self.objects]

    def _inside(self):
        """Per object: its box center is inside the container's walls (with a margin), its lowest corner is below the
        rim (so an object resting on others counts even if it sticks out), and no gripper touches it."""
        c = [it for it in self.items if it["kind"] == "container"][0]
        frame = self._box_frame(c).inv()
        held = set()
        for contact in self._scene.get_contacts() if self._contacts_valid else []:
            names = [body.entity.name for body in contact.bodies]
            if any("gripper" in n for n in names):
                held.update(names)
        hx, hy, hz = c["size"] / 2
        out = []
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) / 2
        for o in self.objects:
            tf = (frame * self._box_frame(o)).to_transformation_matrix()
            p, lowest = tf[:3, 3], ((corners * o["size"]) @ tf[:3, :3].T + tf[:3, 3])[:, 2].min()
            out.append(bool(abs(p[0]) < hx - INSIDE_MARGIN and abs(p[1]) < hy - INSIDE_MARGIN and lowest < hz
                            and o["key"] not in held))
        return out

    def _get_info(self):
        if not self.items:
            return dict(success=False, inside=[], packed=0)
        inside = self._inside()
        return dict(success=all(inside), inside=inside, packed=sum(inside))

    def _check_termination(self):
        return bool(self._get_info()["success"])

    def _get_reward(self):
        return float(self._get_info()["success"])

    @property
    def language_instruction(self):
        prompts = self.prompts
        if not prompts:
            return ""
        if self.target is not None:
            return prompts[self.target]
        return prompts[next((i for i, x in enumerate(self._get_info()["inside"]) if not x), len(prompts) - 1)]


if __name__ == "__main__":
    import sys
    generate_layouts([int(s) for s in sys.argv[1:]] or None)
