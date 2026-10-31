import json

import numpy as np
import sapien

from galaxea_sim import ASSETS_DIR
from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv


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


class CubeBowlPlaceEnv(RoboTwinBaseEnv):
    """Pick up a red cube from the table and place it in a bowl (no scripted solution)."""
    cube_half_size = 0.02
    # measured from 002_bowl base1.glb at model_data1 scale
    bowl_rim_height = 0.047
    bowl_inner_radius = 0.050
    # a cube lying flat on the bowl floor stands ~0.2 cm above the rim
    rim_tolerance = 0.01

    # Placement in the tabletop frame. R1 Lite's head camera only sees the far part of
    # the table, so its objects sit 4 cm farther out to stay fully in frame.
    @property
    def placement(self):
        if self.robot_name == "r1_lite":
            return dict(bowl_x=(-0.02, 0.03), bowl_y=0.06, cube_x=(-0.05, 0.04))
        return dict(bowl_x=(-0.22, -0.16), bowl_y=0.0, cube_x=(-0.12, -0.01))

    def _setup_bowl(self):
        self.side = np.random.choice([-1, 1])  # +1: cube (and bowl offset) on the robot's left
        self.bowl_xy = np.array([np.random.uniform(*self.placement["bowl_x"]), self.side * self.placement["bowl_y"]])
        bowl_pose = rand_pose(
            xlim=[self.bowl_xy[0]],
            ylim=[self.bowl_xy[1]],
            zlim=[0.005],
            rotate_rand=False,
            qpos=[0.707, 0.707, 0, 0],
        )
        self.bowl, self.bowl_data = create_bowl(self._scene, bowl_pose, self.tabletop_center_in_world)

    def _rand_cube_pose(self):
        return rand_pose(
            xlim=list(self.placement["cube_x"]),
            ylim=[-0.25, 0.25],
            zlim=[self.cube_half_size],
            qpos=[0.27, 0.27, 0.65, 0.65],
            rotate_rand=True,
            rotate_lim=[0, 0.1, 0],
        )

    def _setup_cube(self):
        cube_pose = self._rand_cube_pose()
        while self.side * cube_pose.p[1] < 0.08 or np.linalg.norm(cube_pose.p[:2] - self.bowl_xy) < 0.12:
            cube_pose = self._rand_cube_pose()
        cube_pose.set_p(cube_pose.p + self.tabletop_center_in_world)
        self.cube = create_box(
            scene=self._scene,
            pose=cube_pose,
            half_size=(self.cube_half_size,) * 3,
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
            self.bowl.set_pose(sapien.Pose(reset_info["bowl_pose"][:3], reset_info["bowl_pose"][3:]))
            self.cube.set_pose(sapien.Pose(reset_info["cube_pose"][:3], reset_info["cube_pose"][3:]))

    def _get_info(self):
        """Success: every cube corner inside the bowl's inner wall and below its rim, cube released."""
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * self.cube_half_size
        cube_tf = self.cube.get_pose().to_transformation_matrix()
        corners = corners @ cube_tf[:3, :3].T + cube_tf[:3, 3]
        bowl_p = self.bowl.get_pose().p
        inside = np.all(np.linalg.norm(corners[:, :2] - bowl_p[:2], axis=1) < self.bowl_inner_radius)
        below_rim = np.all(corners[:, 2] < bowl_p[2] + self.bowl_rim_height + self.rim_tolerance)
        return dict(success=bool(inside and below_rim and not self._cube_grasped()))

    def _cube_grasped(self):
        for contact in self._scene.get_contacts():
            names = [body.entity.name for body in contact.bodies]
            if "cube" in names and any("gripper" in n for n in names):
                return True
        return False

    def _get_reset_info(self):
        return dict(
            bowl_pose=np.concatenate([self.bowl.get_pose().p, self.bowl.get_pose().q]),
            cube_pose=np.concatenate([self.cube.get_pose().p, self.cube.get_pose().q]),
        )

    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])

    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0

    @property
    def language_instruction(self):
        return "Pick up the cube from the tabletop surface and place it in the bowl."

    def get_object_dict(self):
        return dict(
            bowl=np.concatenate([self.bowl.get_pose().p, self.bowl.get_pose().q]),
            cube=np.concatenate([self.cube.get_pose().p, self.cube.get_pose().q]),
        )
