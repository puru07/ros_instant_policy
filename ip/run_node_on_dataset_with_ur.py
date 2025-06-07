#!/usr/bin/env python3
'''
ROS node for Instant Policy deployment.
'''

import sys
sys.path.insert(0, '/home/mcqueen/anaconda3/envs/ip_env/lib/python3.10/site-packages')  # Adjust this path

import pickle
import time
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
from moveit_msgs.srv import GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory


from rclpy.action import ActionClient
from std_msgs.msg import Float32MultiArray

from tf2_ros import Buffer, TransformListener


class InstantPolicyNode(Node):
    def __init__(self):
        super().__init__('instant_policy_node')
        
        # Initialize transforms dictionary with default values
        self.transforms = {
            'timestamps': None,
            'matrices': None,
            'translations': None,
            'rotations': None
        }
        
        self.rate = self.create_rate(0.2)  # 0.2 Hz = 5 seconds


        # Initialize parameters
        self.declare_parameter('model_path', './checkpoints')
        self.declare_parameter('demo_base_dir', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ip', 'assets', 'rs_data'))
        self.declare_parameter('live_base_dir', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ip', 'assets', 'rs_data'))
        
        self.model_path = self.get_parameter('model_path').value
        self.demo_base_dir = self.get_parameter('demo_base_dir').value
        self.live_base_dir = self.get_parameter('live_base_dir').value
        
        # Initialize MoveIt services
        self.plan_though_pose_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.get_logger().info("Waiting for MoveIt services...")
        self.plan_though_pose_client.wait_for_service()
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

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'

        self.get_logger().info("Instant Policy Node initialized")
        self.timer_tool0 = self.create_timer(0.1, self.get_tool0_pose)   # 10 Hz
        # self.timer_process_data = self.create_timer(1, self.process_data)  
        self.iter = 0
        self.process_data()

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
    
    def get_tool0_pose(self):
        """Get the current pose of tool0 relative to base_link using the latest available transform."""
        try:
            # Wait for transform to be available
            # Log available frames to debug
            self.tf_buffer.can_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            # Get the transform
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            print('updated the transform')
            translation = trans.transform.translation
            rotation = trans.transform.rotation

            # Create transformation matrix
            T = np.eye(4)
            T[:3, 3] = [translation.x, translation.y, translation.z]
            T[:3, :3] = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()

            # Store transform data
            timestamp = trans.header.stamp.sec + trans.header.stamp.nanosec * 1e-9
            self.transforms['timestamps'] = timestamp
            self.transforms['matrices'] = T
            self.transforms['translations'] = [translation.x, translation.y, translation.z]
            self.transforms['rotations'] = [rotation.x, rotation.y, rotation.z, rotation.w]

            return self.transforms
        except Exception as e:
            self.get_logger().warn(f'Could not get tool0 pose: {str(e)}')
            # Return None but keep the transforms dictionary initialized
            return None
 
    def process_data(self):
        """Process the loaded data through the model."""

        while rclpy.ok():
            if self.iter == len(self.live_sample):
                self.get_logger().info("Finished processing live data")
                return None, None
            try:
                # Debug logging to check data structure
                self.get_logger().info(f"Demo sample keys: {self.demo_sample.keys()}")

                full_sample = {
                    'demos': [self.demo_sample],
                    'live': dict(),
                }
                execution_horizon = 8  # should be same as the number of actions in the output of the model
                
                # Get current tool0 pose
                # tool0_pose = self.get_tool0_pose()
                # while tool0_pose is None or self.transforms['matrices'] is None:
                #     tool0_pose = self.get_tool0_pose()
                #     self.get_logger().info("Waiting for tool0 pose...")
                #     time.sleep(0.25)
                
                if self.transforms is None or self.transforms['matrices'] is None:
                    self.get_logger().error("Failed to get tool0 pose, cannot proceed with processing")
                    rclpy.spin_once(self)
                    continue
                else:
                    self.get_logger().info("Got tool0 pose, proceeding with processing")

                i = self.iter
                live_data = self.live_sample[i]
                T_w_e = self.transforms['matrices']
                pcd = live_data['pcds'][0] if isinstance(live_data['pcds'], list) else live_data['pcds']

                full_sample['live']['obs'] = [transform_pcd(subsample_pcd(pcd), np.linalg.inv(T_w_e))]
                full_sample['live']['grips'] = live_data['grips']
                full_sample['live']['actions_grip'] = [np.zeros(8)]
                full_sample['live']['T_w_es'] = [T_w_e]
                full_sample['live']['actions'] = [full_sample['live']['T_w_es'][0].reshape(1, 4, 4).repeat(8, axis=0)]
                
                data = save_sample(full_sample, None)
                print('enconding started')
                if i == 0:
                    demo_scene_node_embds, demo_scene_node_pos = self.model.model.get_demo_scene_emb(
                        data.to(self.model.config['device']))
                
                data.live_scene_node_embds, data.live_scene_node_pos =\
                    self.model.model.get_live_scene_emb(data.to(self.model.config['device']))
                
                data.demo_scene_node_embds = demo_scene_node_embds.clone()
                data.demo_scene_node_pos = demo_scene_node_pos.clone()
                print('AI model is running')
                with torch.no_grad():
                    with torch.autocast(dtype=torch.float32, device_type=self.model.config['device']):
                        actions, grips = self.model.test_step(data.to(self.model.config['device']), 0)
                    actions = actions.squeeze().cpu().numpy()
                    grips = grips.squeeze().cpu().numpy()
                
                self.get_logger().info(f"Predicted actions shape: {actions.shape}")
                self.get_logger().info(f"Predicted grips shape: {grips.shape}")
                # print(actions)

                # exceuting the actions
                for j in range(execution_horizon):
                    # Print current position coordinates
                    print(f"Current position (x, y, z): {self.transforms['translations'][0]:.3f}, {self.transforms['translations'][1]:.3f}, {self.transforms['translations'][2]:.3f}")

                    # pose_mat = T_w_e @ actions[j] # do we need to do this?
                    print(actions[j])
                    pose_mat = actions[j]
                    pose = self.create_pose(pose_mat)
                    
                    # Calculate distance between current position and target pose
                    current_pos = np.array(self.transforms['translations'])
                    target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                    distance = np.linalg.norm(current_pos - target_pos)
                    print(f"Distance to target: {distance:.3f} meters")

                    cartesian_planning = True
                    print(f"Requested position (x, y, z): {pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}")
                    
                    if cartesian_planning:
                        print('cartesian planning')
                        request = self.createCartesiaRequest()
                        request.waypoints.append(pose)
                        print('sending the request to planner')
                        plan_future = self.cartesian_client.call_async(request)

                        print('waiting for the planner to finish')

                        plan_future = self.cartesian_client.call_async(request)
                        self.get_logger().info("Sleeping for 5 seconds...")
                        # time.sleep(5.0)
                        print('spinning until future complete')
                        rclpy.spin_until_future_complete(self, plan_future)
                        print('done sleeping')

                        if not plan_future.result():
                            self.get_logger().warn("Cartesian planning failed.")
                            continue
                        
                        # Sleep for 10 seconds

                    else:
                        print('pose based motion planning')
                        pose_msg = self.transform_to_pose_stamped(pose)
                        print(pose_msg.pose.position)

                        request = self.build_motion_plan_request(pose_msg)
                        plan_future = self.plan_though_pose_client.call_async(request)

                    # Check if the future is done after timeout or normal completion
                    print('checking if the future is done')

                    goal_msg = ExecuteTrajectory.Goal()
                    goal_msg.trajectory = plan_future.result().solution
                    # print('seding the goal to planner')
                    send_goal_future = self.execute_client.send_goal_async(goal_msg)
                    # print('waiting for it to get finished')
                    rclpy.spin_until_future_complete(self, send_goal_future)
                    # result_future = send_goal_future.result().get_result_async()
                    # rclpy.spin_until_future_complete(self, result_future)


                self.iter += 1
                

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
        # pose.position.x = float(pose_mat[0, 3])
        # pose.position.y = float(pose_mat[1, 3] )
        # pose.position.z = float(pose_mat[2, 3])         pose.position.x = float(pose_mat[0, 3])
        pose.position.x = self.transforms['translations'][0] + 0.05
        pose.position.y = self.transforms['translations'][1] 
        pose.position.z = self.transforms['translations'][2] 


        # print(f"created pose: {round(pose.position.x,3)} , {round(pose.position.y,3)} , {round(pose.position.z,3)}")
        #print(f"transformed pose: {round(pose.position.x,3)} , {round(pose.position.y,3)} , {round(pose.position.y,3)}")
        # quat = R.from_matrix(pose_mat[:3, :3]).as_quat()
        # pose.orientation.x = quat[0]
        # pose.orientation.y = quat[1]
        # pose.orientation.z = quat[2]
        # pose.orientation.w = quat[3]

        pose.orientation.x = self.transforms['rotations'][0]
        pose.orientation.y = self.transforms['rotations'][1]
        pose.orientation.z = self.transforms['rotations'][2]
        pose.orientation.w = self.transforms['rotations'][3]
        return pose

    def transform_to_pose_stamped(self, pose):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose = pose
        return pose_msg

    def build_motion_plan_request(self, pose):
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = 'ur_manipulator'
        request.motion_plan_request.allowed_planning_time = 10.0 # 5.0
        request.motion_plan_request.start_state.is_diff = True

        pos_constraint = PositionConstraint()
        pos_constraint.header = pose.header
        pos_constraint.link_name = "tool0"
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        region = SolidPrimitive()
        region.type = SolidPrimitive.BOX
        # region.dimensions = [0.001, 0.001, 0.001]
        region.dimensions = [0.1, 0.1, 0.1]

        bounding_volume = BoundingVolume()
        bounding_volume.primitives.append(region)
        bounding_volume.primitive_poses.append(pose.pose)
        pos_constraint.constraint_region = bounding_volume
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header = pose.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = pose.pose.orientation
        # ori_constraint.absolute_x_axis_tolerance = 0.05 #0.01
        # ori_constraint.absolute_y_axis_tolerance = 0.05 #0.01
        # ori_constraint.absolute_z_axis_tolerance = 0.05 #0.01
        # ori_constraint.weight = 1.0
        ori_constraint.absolute_x_axis_tolerance = 0.1  # Relax tolerance
        ori_constraint.absolute_y_axis_tolerance = 0.1  # Relax tolerance
        ori_constraint.absolute_z_axis_tolerance = 0.1  # Relax tolerance
        ori_constraint.weight = 1.0  # Keep the same weight for the orientation constraint

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        request.motion_plan_request.goal_constraints.append(constraints)
        
        # no constraints
        # request.motion_plan_request.goal_constraints = []
        return request    

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