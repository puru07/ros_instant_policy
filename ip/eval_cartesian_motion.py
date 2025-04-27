import sys

sys.path.insert(0, '/home/mcqueen/anaconda3/envs/ip_env/lib/python3.10/site-packages')  # Adjust this path

# print("Python executable:", sys.executable)
# print("sys.path:", sys.path)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
from ip.utils.common_utils import *
from ip.utils.data_proc import *
from ip.utils.rl_bench_tasks import TASK_NAMES
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from tqdm import trange, tqdm
import argparse
import pickle
from ip.models.diffusion import GraphDiffusion

class RolloutPoseNode(Node):
    def __init__(self, model, task_name, num_demos, num_rollouts, num_traj_wp, restrict_rot):
        super().__init__('rollout_pose_node')
        self.model = model
        self.task_name = task_name
        self.num_demos = num_demos
        self.num_rollouts = num_rollouts
        self.restrict_rot = restrict_rot
        self.num_traj_wp = num_traj_wp

        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.get_logger().info("Waiting for MoveIt services...")
        self.cartesian_client.wait_for_service()
        self.execute_client.wait_for_server()
        self.get_logger().info("Services ready.")

        self.run_rollouts()

    def run_rollouts(self, execution_horizon=8, headless=False):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        env = Environment(MoveArmThenGripper(EndEffectorPoseViaIK(), Discrete()), './', obs_config, headless=headless)
        env.launch()
        task = env.get_task(TASK_NAMES[self.task_name])

        # for planning of robot arm in rl bench
        def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
                distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
            return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                                relative_to=relative_to)

        env._robot.arm.get_path = temp
        env._scene._start_arm_joint_pos = np.array([6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
                                                    -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

        if self.restrict_rot:
            rot_bounds = env._scene.task.base_rotation_bounds()
            mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
            env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                            (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))

        task.reset()
        ####################################################################################################################
        
        full_sample = {'demos': [{} for _ in range(self.num_demos)], 'live': {}}
        for i in tqdm(range(self.num_demos), desc=f'Collecting demos', total=self.num_demos, leave=False):
            done = False
            while not done:
                try:
                    # task.set_variation(i % 2 * 2)
                    demos = task.get_demos(1, live_demos=True, max_attempts=1000)  # -> List[List[Observation]]
                    sample = rl_bench_demo_to_sample(demos[0])
                    full_sample['demos'][i] = sample_to_cond_demo(sample, self.num_traj_wp)
                    assert len(full_sample['demos'][i]['obs']) == self.num_traj_wp
                    done = True
                except:
                    continue



        # chatgpt wrote the for loop below, to replace the for loop above
        # for i in range(self.num_demos):
        #     demos = task.get_demos(1, live_demos=True)
        #     sample = rl_bench_demo_to_sample(demos[0])
        #     full_sample['demos'][i] = sample_to_cond_demo(sample, 10)
        successes = []
        for i in trange(self.num_rollouts):
            task.reset()
            env_action = np.zeros(8)
            # number of steps in rollouts.

            max_execution_steps = 30
            for k in range(max_execution_steps):
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

                if k == 0:
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

                # 🧭 Plan a full cartesian path over the predicted poses
                request = self.createCartesiaRequest()

                for j in range(execution_horizon):
                    pose_mat = T_w_e @ actions[j]
                    pose = self.create_pose(pose_mat)
                    print(pose.position)
                    request.waypoints.append(pose)

                    env_action[:7] = transform_to_pose(T_w_e @ actions[j])
                    env_action[7] = int((grips[j] + 1) / 2 > 0.5)

                    try:
                        curr_obs, reward, terminate = task.step(env_action)
                        success = int(terminate and reward > 0.)
                    except Exception as e:
                        terminate = True
                    if terminate:
                        break
                    
                    plan_future = self.cartesian_client.call_async(request)
                    rclpy.spin_until_future_complete(self, plan_future)

                    if not plan_future.result():
                        self.get_logger().warn("Cartesian planning failed.")
                        continue

                    goal_msg = ExecuteTrajectory.Goal()
                    goal_msg.trajectory = plan_future.result().solution
                    print('seding the goal to planner')
                    send_goal_future = self.execute_client.send_goal_async(goal_msg)
                    print('waiting for it to get finished')
                    rclpy.spin_until_future_complete(self, send_goal_future)
                    # result_future = send_goal_future.result().get_result_async()
                    # rclpy.spin_until_future_complete(self, result_future)

                # plan_future = self.cartesian_client.call_async(request)
                # rclpy.spin_until_future_complete(self, plan_future)

                # if not plan_future.result():
                #     self.get_logger().warn("Cartesian planning failed.")
                #     continue

                # goal_msg = ExecuteTrajectory.Goal()
                # goal_msg.trajectory = plan_future.result().solution
                # send_goal_future = self.execute_client.send_goal_async(goal_msg)
                # rclpy.spin_until_future_complete(self, send_goal_future)
                # result_future = send_goal_future.result().get_result_async()
                # rclpy.spin_until_future_complete(self, result_future)

                successes.append(success)
        env.shutdown()
    
    def createCartesiaRequest(self):
            request = GetCartesianPath.Request()
            request.group_name = 'ur_manipulator'
            request.link_name = 'tool0'
            request.max_step = 1.0 # 0.01  # 1cm resolution
            request.jump_threshold = 0.0
            request.avoid_collisions = True
            request.start_state.is_diff = True
            return request
                
    
    def create_pose(self, pose_mat):
        'returns the pose'
        pose = Pose()
        pose.position.x = pose_mat[0, 3]
        pose.position.y = pose_mat[2, 3]- 1.0  # to bring it within the workspace of UR5
        pose.position.z = pose_mat[1, 3]
        quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose
    

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
    parser.add_argument('--num_demos', type=int, default=1) # originally it was 2
    parser.add_argument('--num_rollouts', type=int, default=1)
    parser.add_argument('--restrict_rot', type=int, default=1)
    parser.add_argument('--compile_models', type=int, default=0)
    args = parser.parse_args()

    model_path = './checkpoints'
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
    print('loaded the config file, here it is')
    print(config)
    config['device']='cpu'
    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = args.num_demos
    config['num_diffusion_iters_test'] = 4
    num_traj_wp = config['traj_horizon'] # 10
    
    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=True,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=args.num_demos)
    model.eval()

    if args.compile_models:
        model.model.compile_models()

    rclpy.init()
    node = RolloutPoseNode(model, args.task_name, args.num_demos, args.num_rollouts, num_traj_wp, bool(args.restrict_rot))
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':

    main()
