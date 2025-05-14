#!/usr/bin/env python3

"""
Pose Control Evaluation Script

This script implements a simple pose-based control evaluation for robotic tasks using MoveIt.
It uses kinematic path planning (GetMotionPlan) to execute trajectories in joint space.

Key Features:
1. Uses GetMotionPlan service for kinematic path planning
2. Plans paths in joint space (not Cartesian space)
3. Simple pose transformation for UR5 workspace
4. Basic demo collection without retries
5. Default task: 'plate_out'

Usage:
    ros2 run ip eval_pose_control --ros-args --task_name <task_name> --num_demos <num> --num_rollouts <num>

Example:
    ros2 run ip eval_pose_control --ros-args --task_name plate_out --num_demos 1 --num_rollouts 1
"""

import sys

sys.path.insert(0, '/home/mcqueen/anaconda3/envs/ip_env/lib/python3.10/site-packages')  # Adjust this path

# print("Python executable:", sys.executable)
# print("sys.path:", sys.path)


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation as R

import numpy as np
from ip.utils.common_utils import *
from ip.utils.data_proc import *
from ip.utils.rl_bench_tasks import TASK_NAMES
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from tqdm import trange
import argparse
import pickle
from ip.models.diffusion import GraphDiffusion

class RolloutPoseNode(Node):
    def __init__(self, model, task_name, num_demos, num_rollouts, restrict_rot):
        super().__init__('rollout_pose_node')
        self.model = model
        self.task_name = task_name
        self.num_demos = num_demos
        self.num_rollouts = num_rollouts
        self.restrict_rot = restrict_rot

        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.get_logger().info("Waiting for MoveIt services...")
        self.plan_client.wait_for_service()
        self.execute_client.wait_for_server()
        self.get_logger().info("Services ready.")

        self.run_rollouts()

    def run_rollouts(self, execution_horizon=8, headless = False):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        env = Environment(MoveArmThenGripper(EndEffectorPoseViaIK(), Discrete()), './', obs_config, headless=headless)
        env.launch()
        task = env.get_task(TASK_NAMES[self.task_name])

        if self.restrict_rot:
            rot_bounds = env._scene.task.base_rotation_bounds()
            mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
            env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                            (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))

        task.reset()
        full_sample = {'demos': [{} for _ in range(self.num_demos)], 'live': {}}

        for i in range(self.num_demos):
            demos = task.get_demos(1, live_demos=True)
            sample = rl_bench_demo_to_sample(demos[0])
            full_sample['demos'][i] = sample_to_cond_demo(sample, 10)

        for i in trange(self.num_rollouts):
            task.reset()
            curr_obs = task.get_observation()
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            pcd = transform_pcd(subsample_pcd(get_point_cloud(curr_obs)), np.linalg.inv(T_w_e))
            full_sample['live'] = {
                'obs': [pcd],
                'grips': [curr_obs.gripper_open],
                'actions_grip': [np.zeros(8)],
                'T_w_es': [T_w_e],
                'actions': [T_w_e.reshape(1, 4, 4).repeat(8, axis=0)]
            }
            data = save_sample(full_sample, None)
            device = self.model.config['device']

            if i == 0:
                demo_emb, demo_pos = self.model.model.get_demo_scene_emb(data.to(device))
            live_emb, live_pos = self.model.model.get_live_scene_emb(data.to(device))

            data.live_scene_node_embds = live_emb.clone()
            data.live_scene_node_pos = live_pos.clone()
            data.demo_scene_node_embds = demo_emb.clone()
            data.demo_scene_node_pos = demo_pos.clone()

            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=device):
                    actions, grips = self.model.test_step(data.to(device), 0)
                actions = actions.squeeze().cpu().numpy()
                grips = grips.squeeze().cpu().numpy()

            for j in range(execution_horizon):
                pose_mat = T_w_e @ actions[j]
                pose_msg = self.transform_to_pose_stamped(pose_mat)
                print(pose_msg.pose.position)

                request = self.build_motion_plan_request(pose_msg)
                plan_future = self.plan_client.call_async(request)
                rclpy.spin_until_future_complete(self, plan_future)
                if not plan_future.result() or not plan_future.result().motion_plan_response.trajectory.joint_trajectory.points:
                    self.get_logger().warn("Planning failed.")
                    break

                goal_msg = ExecuteTrajectory.Goal()
                goal_msg.trajectory = plan_future.result().motion_plan_response.trajectory

                send_goal_future = self.execute_client.send_goal_async(goal_msg)
                rclpy.spin_until_future_complete(self, send_goal_future)
                result_future = send_goal_future.result().get_result_async()
                rclpy.spin_until_future_complete(self, result_future)

        env.shutdown()

    def transform_to_pose_stamped(self, T):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = T[0, 3]
        pose_msg.pose.position.y = T[2, 3] - 1.0  # to bring it within the workspace of UR5
        pose_msg.pose.position.z = T[1, 3]
        print(pose_msg.pose.position)
        quat = R.from_matrix(T[:3, :3]).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        return pose_msg

    def build_motion_plan_request(self, pose):
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = 'ur_manipulator'
        request.motion_plan_request.allowed_planning_time = 5.0
        request.motion_plan_request.start_state.is_diff = True

        pos_constraint = PositionConstraint()
        pos_constraint.header = pose.header
        pos_constraint.link_name = "tool0"
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        region = SolidPrimitive()
        region.type = SolidPrimitive.BOX
        region.dimensions = [0.001, 0.001, 0.001]

        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(region)
        bounding_volume.primitive_poses.append(pose.pose)
        pos_constraint.constraint_region = bounding_volume
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header = pose.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.05 #0.01
        ori_constraint.absolute_y_axis_tolerance = 0.05 #0.01
        ori_constraint.absolute_z_axis_tolerance = 0.05 #0.01
        ori_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        request.motion_plan_request.goal_constraints.append(constraints)

        return request

def rl_bench_demo_to_sample(demo):
    sample = {'pcds': [], 'T_w_es': [], 'grips': []}
    for obs in demo:
        pcd = get_point_cloud(obs)
        sample['pcds'].append(pcd)
        sample['T_w_es'].append(pose_to_transform(obs.gripper_pose))
        sample['grips'].append(obs.gripper_open)
    return sample

def get_point_cloud(obs, camera_names=('front', 'left_shoulder', 'right_shoulder')):
    pcds = []
    for name in camera_names:
        pcd = getattr(obs, f'{name}_point_cloud')
        mask = getattr(obs, f'{name}_mask')
        masked_pcd = pcd[mask > 60]
        pcds.append(masked_pcd)
    return downsample_pcd(np.concatenate(pcds, axis=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='plate_out')
    parser.add_argument('--num_demos', type=int, default=1)
    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--restrict_rot', type=int, default=1)
    parser.add_argument('--compile_models', type=int, default=0)
    args = parser.parse_args()

    model_path = './checkpoints'
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
    config['device']='cpu'
    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = args.num_demos
    config['num_diffusion_iters_test'] = 4

    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=True,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=args.num_demos)
    model.eval()

    if args.compile_models:
        model.model.compile_models()

    rclpy.init()
    node = RolloutPoseNode(model, args.task_name, args.num_demos, args.num_rollouts, bool(args.restrict_rot))
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':

    main()


