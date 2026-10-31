import json
import math
from copy import deepcopy

import numpy as np
import sapien

from galaxea_sim import ASSETS_DIR
from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.utils.rand_utils import rand_pose
from .blocks_stack_easy import BlocksStackEasyEnv


def create_bowl(scene: sapien.Scene, pose: sapien.Pose, tabletop_center_in_world: np.ndarray):
    """RoboTwin2.0 002_bowl variant 1: convex hull per piece of its 28-piece collision mesh
    (CoACD fails on the visual mesh), visual from base1.glb."""
    modeldir = ASSETS_DIR / "robotwin_models" / "002_bowl"
    with open(modeldir / "model_data1.json") as f:
        model_data = json.load(f)
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    builder.add_multiple_convex_collisions_from_file(
        filename=(modeldir / "collision_base1.glb").as_posix(),
        scale=model_data["scale"],
        decomposition="none",
    )
    builder.add_visual_from_file(filename=(modeldir / "base1.glb").as_posix(), scale=model_data["scale"])
    bowl = builder.build(name="bowl")
    pose.set_p(pose.get_p() + tabletop_center_in_world)
    bowl.set_pose(pose)
    return bowl, model_data


class CubeBowlPlaceEnv(BlocksStackEasyEnv):
    """Pick up a cube from the table and place it in a bowl.

    Reuses BlocksStackEasyEnv's grasp/place solution; the place target
    (place_pos_x_offset, y ~= 0) is moved to the bowl each reset.
    """
    bowl_rim_height = 0.049  # 002_bowl model_data1 extents[1] * scale
    bowl_inner_radius = 0.045
    # end-effector height above the table when releasing over the bowl
    release_height = 0.17

    @property
    def bowl_x_offset_range(self):
        # R1 Lite's head camera does not see the table edge closest to the robot
        return (-0.08, -0.02) if self.robot_name == "r1_lite" else (-0.22, -0.16)

    @property
    def bowl_y_offset(self):
        # R1 Lite cannot reliably reach y=0 that far out; shift the bowl toward the working arm
        return 0.06 if self.robot_name == "r1_lite" else 0.0

    def _setup_bowl(self):
        self.side = np.random.choice([-1, 1])  # +1: cube/working arm on the left
        self.place_pos_x_offset = np.random.uniform(*self.bowl_x_offset_range)
        bowl_pose = rand_pose(
            xlim=[self.place_pos_x_offset],
            ylim=[self.side * self.bowl_y_offset],
            zlim=[0.005],
            rotate_rand=False,
            qpos=[0.707, 0.707, 0, 0],
        )
        self.bowl, self.bowl_data = create_bowl(self._scene, bowl_pose, self.tabletop_center_in_world)

    def _setup_cube(self):
        rand_pos = self._rand_pose()
        bowl_xy = np.array([self.place_pos_x_offset, self.side * self.bowl_y_offset])
        while self.side * rand_pos.p[1] < 0.08 \
            or np.linalg.norm(rand_pos.p[:2] - bowl_xy) < 0.12:
            rand_pos = self._rand_pose()
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.cube = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(1, 0, 0),
            name="cube"
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "bowl"):
            self._scene.remove_actor(self.bowl)
        if hasattr(self, "cube"):
            self._scene.remove_actor(self.cube)
        self._setup_bowl()
        self._setup_cube()
        if reset_info is not None:
            self.place_pos_x_offset = reset_info["bowl_pose"][0] - self.tabletop_center_in_world[0]
            self.bowl.set_pose(sapien.Pose(reset_info["bowl_pose"][:3], reset_info["bowl_pose"][3:]))
            self.cube.set_pose(sapien.Pose(reset_info["cube_pose"][:3], reset_info["cube_pose"][3:]))

    def solution(self):
        cube_p = self.cube.get_pose().p
        arm = 'right' if cube_p[1] < 0 else 'left'
        # top-down grasp aligned with the cube's yaw, as in BlocksStackEasyEnv.move_block
        cube_yaw = math.fmod(self.cube.get_pose().get_rpy()[2], math.pi / 2)
        grasp_yaw = cube_yaw if arm == 'right' else cube_yaw - math.pi / 2
        (grasp_q := sapien.Pose()).set_rpy(rpy=np.array([0, 0, grasp_yaw]) + self.robot.right_ee_rpy_offset)
        grasp_q = self.rot_down_grip_pose(grasp_q).q.tolist()

        # release orientation as in move_block's place target (works for both R1 Lite and R1 Pro)
        place_yaw = math.pi / 2 if arm == 'right' else -math.pi / 2
        (place_q := sapien.Pose()).set_rpy(rpy=np.array([0, 0, place_yaw]) + self.robot.right_ee_rpy_offset)
        place_q = self.rot_down_grip_pose(place_q).q.tolist()

        pre_grasp = self.tf_to_grasp(list(cube_p + [0, 0, 0.2]) + grasp_q)
        grasp = deepcopy(pre_grasp)
        grasp[2] -= 0.15
        # low release so R1 Lite can reach; the cube still clears the rim by ~2 cm
        bowl_p = self.bowl.get_pose().p
        above_bowl = self.tf_to_grasp([bowl_p[0], bowl_p[1], self.table_height + self.release_height] + place_q)

        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pre_grasp)})
        yield ("open_gripper", {"action_mode": arm})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(grasp)})
        yield ("close_gripper", {"action_mode": arm})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pre_grasp)})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(above_bowl)})
        yield ("open_gripper", {"action_mode": arm})
        yield ("move_to_pose", {f"{arm}_pose": self.robot.left_init_ee_pose if arm == 'left' else self.robot.right_init_ee_pose})

    def _get_info(self):
        cube_p = self.cube.get_pose().p
        bowl_p = self.bowl.get_pose().p
        in_xy = np.linalg.norm(cube_p[:2] - bowl_p[:2]) < self.bowl_inner_radius
        below_rim = cube_p[2] < self.table_height + self.bowl_rim_height + self.block_half_size
        return dict(success=bool(in_xy and below_rim))

    def _get_reset_info(self):
        return dict(
            bowl_pose=np.concatenate([self.bowl.get_pose().p, self.bowl.get_pose().q]),
            cube_pose=np.concatenate([self.cube.get_pose().p, self.cube.get_pose().q]),
        )

    @property
    def language_instruction(self):
        return "Pick up the cube from the tabletop surface and place it in the bowl."

    def get_object_dict(self):
        return dict(
            bowl=np.concatenate([self.bowl.get_pose().p, self.bowl.get_pose().q]),
            cube=np.concatenate([self.cube.get_pose().p, self.cube.get_pose().q]),
        )
