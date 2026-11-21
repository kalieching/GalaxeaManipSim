"""Scripted pick-and-place for KittingEnv, planned with mplib (the BimanualPlanner the layout reach checks use).

Drop-in for G05Policy in collect_kitting.py (--policy scripted): reset(env) plans for env.target, act() serves one
joint command per step and returns None for the action once the script is done or a plan fails. Each attempt: bring
both arms back to the start pose, pick the target with the arm on its side (fingers down or tilted, closing across
a side of the object that fits, a random grasp among those that pass IK), lift it, carry it over a random free spot in the
container, lower it to just above the rim, release and back off upward. Every attempt draws new random choices, so a
failed one is retried differently.
"""
import numpy as np
import transforms3d as t3d
from mplib.pymp import Pose

from galaxea_sim.planners.bimanual import BimanualPlanner
from galaxea_sim.utils.sapien_utils import get_link_by_name

PAD = 0.037           # end-effector link to the middle of the finger pads, m (pads reach 0.7 to 6.7 cm out)
MAX_WIDTH = 0.09      # widest graspable side, m (fingers open 2 x 5 cm)
PRE_GRASP = 0.10      # approach distance, m
LIFT = 0.10           # m
CLEAR = 0.04          # object bottom above the rim while carrying, m
RELEASE = 0.01        # object bottom above the rim at release, m
HOME_SPEED = 1.0      # rad/s, joint-space return to the start pose
GRIP_STEPS = 20
TILTS = (0, 30, 45, 60, 75, 90)  # deg from fingers down (the R1 Pro reaches the table best with level fingers)
SETTLE_STEPS = 15
ARRIVED = 0.01        # rad, arm joints this close to the end of a move count as arrived
MAX_WAIT = 45         # steps to wait for that
GRID = 0.01           # obstacle point spacing, m
LINE_STEP = 0.01      # m per control step on straight-line moves


def _qz(angle):
    return np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])


def _box_points(frame, size, top=True):
    """Points on the faces of a box (sapien pose of its center, size), GRID apart. top=False leaves it open."""
    pts = []
    for axis in range(3):
        for sign in (-1, 1):
            if axis == 2 and sign == 1 and not top:
                continue
            u, v = [i for i in range(3) if i != axis]
            gu, gv = (np.linspace(-size[i] / 2, size[i] / 2, max(2, int(size[i] / GRID) + 1)) for i in (u, v))
            face = np.zeros((len(gu) * len(gv), 3))
            face[:, u], face[:, v] = np.repeat(gu, len(gv)), np.tile(gv, len(gu))
            face[:, axis] = sign * size[axis] / 2
            pts.append(face)
    tf = frame.to_transformation_matrix()
    return np.concatenate(pts) @ tf[:3, :3].T + tf[:3, 3]


class ScriptedPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.planner, self.queue, self.stages, self.approach = None, [], iter(()), {}

    def reset(self, env):
        if self.planner is None:
            self.planner = BimanualPlanner(
                urdf_path=f"{env.robot.name}/robot.urdf", srdf_path=None, left_arm_move_group=env.left_ee_link_name,
                right_arm_move_group=env.right_ee_link_name, active_joint_names=env.active_joint_names,
                control_freq=env.control_freq)
        self.queue, self.stages, self.last, self.waited, self.stage = [], self._stages(env), None, 0, ""

    def act(self, env, prompt=None):
        """(env action or None when finished, None (no model inputs), {} (no model output))."""
        if not self.queue and self.last is not None and self.waited < MAX_WAIT:
            # the R1 Pro's arms trail a planned trajectory by up to ~0.1 rad: hold its end until they arrive
            qpos = env.robot.get_qpos()
            arms = np.concatenate([qpos[env.left_arm_joint_indices], qpos[env.right_arm_joint_indices]])
            n = len(env.left_arm_joint_indices)
            cmd = np.concatenate([self.last[:n], self.last[n + 1:-1]])
            if np.abs(cmd - arms).max() > ARRIVED:
                self.waited += 1
                return self.last, None, {}
        while not self.queue:
            traj = next(self.stages, None)
            if traj is None:
                return None, None, {}
            self.queue, self.waited = list(traj), 0
        self.last = self.queue.pop(0)
        return self.last, None, {}

    # ---- planning ----
    def _solve(self, env, method, **kw):
        return self.planner.solve((method, kw), env.robot.get_qpos(), env.last_gripper_cmd)

    def _move(self, env, side, p, q):
        """Collision-free move (RRT) of one hand, the torso held still."""
        return self._solve(env, "move_to_pose", **{f"{side}_pose": np.concatenate([p, q])})

    def _line(self, env, side, p, q):
        """Straight-line move of one hand, IK every LINE_STEP with the torso held still (mplib's screw motion would
        move the torso too, which the arm-only commands then leave out). None if IK fails on the way."""
        pl = self.planner
        arm, mask = (pl.left_arm_planner, pl.left_arm_planner_mask) if side == "left" else \
            (pl.right_arm_planner, pl.right_arm_planner_mask)
        qpos, grip = env.robot.get_qpos(), env.last_gripper_cmd
        start = getattr(env.robot, f"{side}_ee_link").entity_pose.p
        cur, traj = pl.sim2mplib_mapping(qpos), []
        for s in np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(p - start) / LINE_STEP))) + 1)[1:]:
            status, sol = arm.IK(Pose(p=start + s * (p - start), q=q), cur, mask=mask, return_closest=True)
            if status != "Success":
                return None
            cur = sol
            left = cur[~pl.left_arm_planner_mask] if side == "left" else qpos[env.left_arm_joint_indices]
            right = cur[~pl.right_arm_planner_mask] if side == "right" else qpos[env.right_arm_joint_indices]
            traj.append(np.concatenate([left, [grip[0]], right, [grip[1]]]))
        return traj

    def _arms(self):
        return self.planner.left_arm_planner, self.planner.right_arm_planner

    def _set_obstacles(self, env):
        """Table top, container (open box) and the other objects as a point cloud for both arms. The target is a
        separate cloud, switched with _target_obstacle."""
        x, y = np.meshgrid(np.arange(0.25, 0.85, GRID), np.arange(-0.6, 0.6, GRID))
        pts = [np.stack([x.ravel(), y.ravel(), np.full(x.size, env.table_top_z)], axis=1)]
        target = env.objects[env.target]
        pts += [_box_points(env._box_frame(it), it["size"], top=it["kind"] != "container")
                for it in env.items if it is not target]
        self.target_points = _box_points(env._box_frame(target), target["size"])
        for arm in self._arms():
            arm.update_point_cloud(np.concatenate(pts), resolution=GRID / 2)

    def _target_obstacle(self, on):
        for arm in self._arms():
            if on:
                arm.update_point_cloud(self.target_points, resolution=GRID / 2, name="target")
            else:
                arm.remove_point_cloud("target")

    def _ik_ok(self, env, side, p, q):
        pl = self.planner
        arm = pl.left_arm_planner if side == "left" else pl.right_arm_planner
        mask = pl.left_arm_planner_mask if side == "left" else pl.right_arm_planner_mask
        return arm.IK(Pose(p=p, q=q), pl.sim2mplib_mapping(env.robot.get_qpos()), mask=mask, n_init_qpos=6)[0] == "Success"

    def _home(self, env):
        qpos, home = env.robot.get_qpos(), np.asarray(env.init_qpos, float)
        arms = list(env.left_arm_joint_indices) + list(env.right_arm_joint_indices)
        n = int(np.ceil(np.abs(home[arms] - qpos[arms]).max() * env.control_freq / HOME_SPEED))
        if n < 2:
            return None
        grip = env.last_gripper_cmd
        return [np.concatenate([qpos[env.left_arm_joint_indices] * (1 - s) + home[env.left_arm_joint_indices] * s,
                                [grip[0]],
                                qpos[env.right_arm_joint_indices] * (1 - s) + home[env.right_arm_joint_indices] * s,
                                [grip[1]]])
                for s in np.linspace(0, 1, n)[1:]]

    def _grasps(self, env, obj):
        """Candidate (side, end-effector position, quaternion, approach direction), in a random order, near arm first."""
        frame = env._box_frame(obj)
        rot = t3d.quaternions.quat2mat(frame.q)
        size = obj["size"]
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        # directions to close the fingers along: each horizontal side narrow enough to grip
        across = [yaw + k * np.pi / 2 for k in range(2) if size[k] <= MAX_WIDTH]
        if not across:
            return []
        round_ = abs(size[0] - size[1]) < 0.01
        bottom, top = frame.p[2] - size[2] / 2, frame.p[2] + size[2] / 2
        near = "left" if frame.p[1] > 0 else "right"
        out = []
        for side in (near, "right" if near == "left" else "left"):
            cands = []
            for tilt, q0 in zip(TILTS, env._grasp_quats(side, TILTS)):
                # the fingers close along world y at phi = 0. Turn the hand about z to close across a side that fits
                # (every 30 deg for a can or bottle). IK sorts out which of these the arm can do.
                phis = np.deg2rad(range(-90, 90, 30)) if round_ else \
                    [c % np.pi - np.pi / 2 + k * np.pi for c in across for k in (0, 1)]
                low = bottom + (0.02 if tilt < 60 else 0.035)  # level fingers need room above the table
                for phi in phis:
                    q = t3d.quaternions.qmult(_qz(phi), q0)
                    a = t3d.quaternions.quat2mat(q) @ self._local_approach(env, side)
                    # between the upper part and, on tall objects, lower down (keeps the hand low enough to
                    # carry it over the rim)
                    h = self.rng.uniform(max(low, min(top - 0.05, bottom + 0.4 * size[2])), max(low, top - 0.025))
                    grasp = np.array([frame.p[0], frame.p[1], h]) - a * PAD
                    cands.append((side, grasp, q, a))
            self.rng.shuffle(cands)
            out += cands
        return out

    def _drop_pose(self, env, side, obj, q, hand):
        """Hand pose (position above the rim, quaternion) that holds the object upright over a free spot in the
        container: random spots, the hand turned about z by up to 60 deg, the first one IK can reach (both the carry
        height and the release height). hand: where the hand is (or will be) while holding the object at its current
        pose. None if none can be reached."""
        frame = env._box_frame(obj)
        offset = t3d.quaternions.quat2mat(q).T @ (hand - (frame.p - [0, 0, obj["size"][2] / 2]))  # in the hand frame
        c = [it for it in env.items if it["kind"] == "container"][0]
        cf = env._box_frame(c)
        rim = cf.p[2] + c["size"][2] / 2
        room = np.maximum(c["size"][:2] / 2 - np.max(obj["size"][:2]) / 2 - 0.02, 0)
        for turn in self.rng.permutation(np.deg2rad([0, 15, -15, 30, -30, 45, -45, 60, -60]).reshape(-1, 1))[:, 0]:
            q2 = t3d.quaternions.qmult(_qz(turn), q)
            for spot in cf.p[:2] + self.rng.uniform(-room, room, (4, 2)):
                above = np.array([*spot, rim + CLEAR]) + t3d.quaternions.quat2mat(q2) @ offset
                if self._ik_ok(env, side, above, q2) and self._ik_ok(env, side, above - [0, 0, CLEAR - RELEASE], q2):
                    return above, q2
        return None

    def _local_approach(self, env, side):
        """Direction the fingers point, in the end-effector frame (differs between the R1 Lite and R1 Pro)."""
        if side not in self.approach:
            ee = getattr(env.robot, f"{side}_ee_link").entity_pose
            f = np.mean([get_link_by_name(env.robot.links, f"{side}_gripper_finger_link{i}").entity_pose.p
                         for i in (1, 2)], axis=0) - ee.p
            self.approach[side] = t3d.quaternions.quat2mat(ee.q).T @ (f / np.linalg.norm(f))
        return self.approach[side]

    def _stages(self, env):
        """Yields one action trajectory per stage, named in self.stage. Stops at a stage that cannot be planned."""
        self.stage = "home"
        if home := self._home(env):
            yield home
        obj = env.objects[env.target]
        self._set_obstacles(env)
        self.stage = "grasp IK"
        self._target_obstacle(False)  # the fingers end up around the target, but must not sweep through it on the way
        cands = [(side, g, q, g - a * PRE_GRASP) for side, g, q, a in self._grasps(env, obj)
                 if self._ik_ok(env, side, g, q) and self._ik_ok(env, side, g + [0, 0, LIFT], q)]
        self._target_obstacle(True)
        # the first grasp whose approach, lift and some drop pose over the container all have IK solutions
        side, grasp, q, pre = next(((s, g, q, p) for s, g, q, p in cands if self._ik_ok(env, s, p, q)
                                    and self._drop_pose(env, s, obj, q, g) is not None), (None,) * 4)
        if side is None:
            return
        grip = lambda method, steps=GRIP_STEPS: self._solve(env, method, steps=steps, action_mode=side)  # noqa: E731
        self.stage = "open"
        yield grip("open_gripper")
        self.stage = "pre-grasp"
        if (traj := self._move(env, side, pre, q)) is None:
            return
        yield traj
        self.stage = "grasp"
        self._target_obstacle(False)
        if (traj := self._line(env, side, grasp, q)) is None:
            return
        yield traj
        self.stage = "close"
        yield grip("close_gripper")
        self.stage = "lift"
        if (traj := self._line(env, side, grasp + [0, 0, LIFT], q)) is None:
            return
        yield traj
        self.stage = "carry IK"
        hand = getattr(env.robot, f"{side}_ee_link").entity_pose.p
        if (drop := self._drop_pose(env, side, obj, q, hand)) is None:
            return
        above, q = drop
        self.stage = "carry"
        if (traj := self._move(env, side, above, q)) is None:
            return
        yield traj
        self.stage = "lower"
        if (traj := self._line(env, side, above - [0, 0, CLEAR - RELEASE], q)) is None:
            return
        yield traj
        self.stage = "release"
        yield grip("open_gripper")
        self.stage = "back off"
        if (traj := self._line(env, side, above + [0, 0, 0.08], q)) is not None:
            yield traj
        self.stage = "settle"
        yield grip("open_gripper", SETTLE_STEPS)  # hold still while the object settles
