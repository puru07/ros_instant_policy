#!/usr/bin/env python3
'''
ROS node for Instant Policy deployment.
'''

import sys
sys.path.insert(0, '/home/mcqueen/anaconda3/envs/ip_env/lib/python3.10/site-packages')  # Adjust this path

import pickle
from ip.models.diffusion import GraphDiffusion
from ip.utils.data_proc import *
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import torch

#ip stuff
from ip.utils.common_utils import *
from ip.utils.data_proc import *

# ros2 stuff
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
from std_msgs.msg import Float32MultiArray


class InstantPolicyNode(Node):
    def __init__(self):
        super().__init__('instant_policy_node')
        
        # Initialize parameters
        self.declare_parameter('model_path', './checkpoints')
        self.declare_parameter('demo_base_dir', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ip', 'assets', 'rs_data'))
        self.declare_parameter('live_base_dir', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ip', 'assets', 'rs_data'))
        
        self.model_path = self.get_parameter('model_path').value
        self.demo_base_dir = self.get_parameter('demo_base_dir').value
        self.live_base_dir = self.get_parameter('live_base_dir').value
        
        # Initialize MoveIt services
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.get_logger().info("Waiting for MoveIt services...")
        self.cartesian_client.wait_for_service()
        self.execute_client.wait_for_server()
        self.get_logger().info("Services ready.")

        # Initialize model
        self.initialize_model()
        
        # Initialize publishers
        self.predicted_actions_pub = self.create_publisher(Float32MultiArray, '/instant_policy/actions', 1)
        self.predicted_grips_pub = self.create_publisher(Float32MultiArray, '/instant_policy/grips', 1)
        
        # Load demo data
        self.load_demo_data()
        self.load_live_data()


        self.get_logger().info("Instant Policy Node initialized")
        # Create timer to call process_data at 2 Hz
        self.process_data()
        # self.timer = self.create_timer(0.5, self._timer_callback)

    def initialize_model(self):
        """Initialize the Instant Policy model."""
        config = pickle.load(open(f'{self.model_path}/config.pkl', 'rb'))
        
        config['num_layers'] = 2
        config['device'] = 'cpu'
        config['compile_models'] = False
        config['batch_size'] = 1
        config['num_demos'] = 1
        config['num_diffusion_iters_test'] = 4

        self.model = GraphDiffusion.load_from_checkpoint(
            f'{self.model_path}/model.pt', 
            config=config, 
            strict=False,
            map_location=config['device']
        ).to(config['device'])
        
        self.model.model.reinit_graphs(1, num_demos=1)
        self.model.eval()
        
        self.get_logger().info("Model initialized successfully")

    def load_demo_data(self):
        """Load demonstration data (LATEST) from the specified directory."""
        if not os.path.exists(self.demo_base_dir):
            raise ValueError(f"Demo directory does not exist: {self.demo_base_dir}")
        
        demo_folders = glob.glob(os.path.join(self.demo_base_dir, '*'))
        if not demo_folders:
            raise ValueError("No demo folders found in the specified directory")
        
        demo_folders.sort(key=lambda x: os.path.basename(x), reverse=True)
        demo_folder = demo_folders[0]
        
        self.get_logger().info(f"Loading demo from: {os.path.basename(demo_folder)}")
        self.demo_sample = self.load_demo_from_folder(demo_folder)
    
    def load_live_data(self):
        """Load live data (LATEST) from the specified directory."""
        if not os.path.exists(self.live_base_dir):
            raise ValueError(f"Live directory does not exist: {self.live_base_dir}")
        
        live_folders = glob.glob(os.path.join(self.live_base_dir, '*'))
        if not live_folders:
            raise ValueError("No live folders found in the specified directory")
        
        live_folders.sort(key=lambda x: os.path.basename(x), reverse=True)
        live_folder = live_folders[0]
        
        self.get_logger().info(f"Loading live from: {os.path.basename(live_folder)}")
        self.live_sample = self.load_live_from_folder(live_folder)    

    def load_demo_from_folder(self, folder_path):
        """Load demonstration data from a folder containing cropped PCD files and transformation matrices."""
        transform_file = os.path.join(folder_path, 'tool0_transforms.pkl')
        with open(transform_file, 'rb') as f:
            transforms = pickle.load(f)
        
        pcd_dir = os.path.join(folder_path, 'cropped_pcds')
        pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
        
        pcds = []
        for pcd_file in pcd_files:
            with open(pcd_file, 'r') as f:
                lines = f.readlines()
                points = []
                for line in lines[11:]:
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])
                pcds.append(np.array(points))
        
        num_pcds = len(pcds)
        transforms['matrices'] = transforms['matrices'][:num_pcds]
        
        # Create demo sample with both pcds and obs keys
        # demo_sample = {
        #     'pcds': pcds,
        #     'obs': pcds,  # Add obs key with same data
        #     'T_w_es': transforms['matrices'],
        #     'grips': np.zeros(len(pcds))
        # }
        demo_sample = {
            'obs': pcds,  # Add obs key with same data
            'T_w_es': transforms['matrices'],
            'grips': np.zeros(len(pcds))
        }
        self.get_logger().info(f"Loaded demo sample with keys: {demo_sample.keys()}")
        self.get_logger().info(f"Number of point clouds: {len(pcds)}")
        self.get_logger().info(f"Number of transforms: {len(transforms['matrices'])}")
        
        return demo_sample
    
    def load_live_from_folder(self, folder_path):
        """Load live data from a folder containing cropped PCD files and transformation matrices."""
        # Load transformation matrices
        transform_file = os.path.join(folder_path, 'tool0_transforms.pkl')
        with open(transform_file, 'rb') as f:
            transforms = pickle.load(f)
        
        # Load cropped PCD files
        pcd_dir = os.path.join(folder_path, 'cropped_pcds')
        pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
        
        # Create a list of live data dictionaries
        live_data_list = []
        for i, pcd_file in enumerate(pcd_files):
            # Read PCD file
            with open(pcd_file, 'r') as f:
                lines = f.readlines()
                # Skip header
                points = []
                for line in lines[11:]:  # Skip PCD header
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])
                pcd = np.array(points, dtype=np.float64)  # Ensure float64 type
            
            # Create live data dictionary for this point cloud
            live_data = {
                'pcds': [pcd],  # Single point cloud in a list
                'T_w_es': [transforms['matrices'][i]],  # Single transform in a list
                'grips': np.array([0.0])  # Single grip state
            }
            live_data_list.append(live_data)
        
        # Debug: Print live data information
        self.get_logger().info(f"Loaded live data from {folder_path}:")
        self.get_logger().info(f"  Number of live data entries: {len(live_data_list)}")
        self.get_logger().info(f"  Live data keys: {live_data_list[0].keys()}")
        if len(live_data_list) > 0:
            self.get_logger().info(f"  Point cloud shape: {live_data_list[0]['pcds'][0].shape}")
            self.get_logger().info(f"  Point cloud dtype: {live_data_list[0]['pcds'][0].dtype}")
        
        return live_data_list
    '''
    def process_point_cloud(self, pcd_msg):
        """Process incoming point cloud and run inference."""
        try:
            # Convert PointCloud2 to numpy array
            points = []
            for point in pc2.read_points(pcd_msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])
            pcd = np.array(points, dtype=np.float64)
            
            # Get current transform
            try:
                transform = self.tf_buffer.lookup_transform(
                    'world', 'tool0', rospy.Time(0))
                T_w_e = self.transform_to_matrix(transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Failed to get transform: {e}")
                return
            
            # Prepare live data
            live_data = {
                'pcds': [pcd],
                'T_w_es': [T_w_e],
                'grips': np.array([0.0])
            }
            
            # Process through model
            actions, grips = self.process_data(self.model, self.demo_sample, [live_data])
            
            if actions is not None and grips is not None:
                # Publish results
                actions_msg = Float32MultiArray()
                actions_msg.data = actions.flatten().tolist()
                self.predicted_actions_pub.publish(actions_msg)
                
                grips_msg = Float32MultiArray()
                grips_msg.data = grips.flatten().tolist()
                self.predicted_grips_pub.publish(grips_msg)
                
        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {str(e)}")

    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_quat([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]).as_matrix()
        matrix[:3, 3] = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ]
        return matrix
    '''
    def _timer_callback(self):
        """Timer callback to process data at 2 Hz."""
        self.process_data()
        
    def process_data(self):
        """Process the loaded data through the model."""
        try:
            # Debug logging to check data structure
            self.get_logger().info(f"Demo sample keys: {self.demo_sample.keys()}")
            # if 'pcds' in self.demo_sample:
            #     self.get_logger().info(f"Demo sample pcds type: {type(self.demo_sample['pcds'])}")
            #     self.get_logger().info(f"Demo sample pcds length: {len(self.demo_sample['pcds'])}")
            
            # # Check if we need to rename pcds to obs
            # if 'pcds' in self.demo_sample and 'obs' not in self.demo_sample:
            #     self.demo_sample['obs'] = self.demo_sample.pop('pcds')
            
            full_sample = {
                'demos': [self.demo_sample],
                'live': dict(),
            }
            execution_horizon = 8 # should be same as the number of actions in the output of the model

            for i, live_data in enumerate(self.live_sample):
                T_w_e = live_data['T_w_es'][0]
                pcd = live_data['pcds'][0] if isinstance(live_data['pcds'], list) else live_data['pcds']

                full_sample['live']['obs'] = [transform_pcd(subsample_pcd(pcd), np.linalg.inv(T_w_e))]
                full_sample['live']['grips'] = live_data['grips']
                full_sample['live']['actions_grip'] = [np.zeros(8)]
                full_sample['live']['T_w_es'] = [T_w_e]
                full_sample['live']['actions'] = [full_sample['live']['T_w_es'][0].reshape(1, 4, 4).repeat(8, axis=0)]
                
                data = save_sample(full_sample, None)
                
                if i == 0:
                    demo_scene_node_embds, demo_scene_node_pos = self.model.model.get_demo_scene_emb(
                        data.to(self.model.config['device']))
                
                data.live_scene_node_embds, data.live_scene_node_pos =\
                    self.model.model.get_live_scene_emb(data.to(self.model.config['device']))
                
                data.demo_scene_node_embds = demo_scene_node_embds.clone()
                data.demo_scene_node_pos = demo_scene_node_pos.clone()
                
                with torch.no_grad():
                    with torch.autocast(dtype=torch.float32, device_type=self.model.config['device']):
                        actions, grips = self.model.test_step(data.to(self.model.config['device']), 0)
                    actions = actions.squeeze().cpu().numpy()
                    grips = grips.squeeze().cpu().numpy()
                
                self.get_logger().info(f"Predicted actions shape: {actions.shape}")
                self.get_logger().info(f"Predicted grips shape: {grips.shape}")
                print(actions)

                for j in range(execution_horizon):
                    pose_mat = T_w_e @ actions[j]
                    print(f"\nPose matrix for step {j} (T_w_e @ actions[j]):")
                    print(pose_mat)
                    pose = self.create_pose(pose_mat)
                    request = self.createCartesiaRequest()
                    request.waypoints.append(pose)

                    
                    plan_future = self.cartesian_client.call_async(request)
                    rclpy.spin_until_future_complete(self, plan_future)

                    if not plan_future.result():
                        self.get_logger().warn("Cartesian planning failed.")
                        continue

                    goal_msg = ExecuteTrajectory.Goal()
                    goal_msg.trajectory = plan_future.result().solution
                    # print('seding the goal to planner')
                    send_goal_future = self.execute_client.send_goal_async(goal_msg)
                    # print('waiting for it to get finished')
                    rclpy.spin_until_future_complete(self, send_goal_future)
                    # result_future = send_goal_future.result().get_result_async()
                    # rclpy.spin_until_future_complete(self, result_future)


                return actions, grips

        except Exception as e:
            self.get_logger().error(f"Error in processing: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None, None

    def createCartesiaRequest(self):
        request = GetCartesianPath.Request()
        request.group_name = 'ur_manipulator'
        request.link_name = 'tool0'
        request.max_step = 0.01 # 0.01  # 1cm resolution
        request.jump_threshold = 0.0
        request.avoid_collisions = True
        request.start_state.is_diff = True
        return request                  

    def create_pose(self, pose_mat):
        'returns the pose'
        pose = Pose()
        pose.position.y = (pose_mat[0, 3] ) + 0.2
        pose.position.x = -1*(pose_mat[1, 3])
        pose.position.z = (pose_mat[2, 3]- 0.8)  # to bring it within the workspace of UR5
        print(f" pose: {round(pose_mat[0, 3],3)} , {round(pose_mat[1, 3],3)} , {round(pose_mat[2, 3],3)} :::  transformed pose: {round(pose.position.x,3)} , {round(pose.position.y,3)} , {round(pose.position.z,3)}")
        #print(f"transformed pose: {round(pose.position.x,3)} , {round(pose.position.y,3)} , {round(pose.position.y,3)}")
        quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose
    

def main(args=None):
    rclpy.init(args=args)
    node = InstantPolicyNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 