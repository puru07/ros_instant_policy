#!/usr/bin/env python3
'''
This scripts shows and example of how Instant Policy could be used at deployment.
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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import torch
import sensor_msgs_py.point_cloud2 as pc2

class DeploymentNode(Node):
    def __init__(self, model, num_demos, num_traj_wp, num_diffusion_iters, demos):
        super().__init__('deployment_node')
        
        # Store model and parameters
        self.model = model
        self.num_demos = num_demos
        self.num_traj_wp = num_traj_wp
        self.num_diffusion_iters = num_diffusion_iters
        self.demos = demos  # Store the loaded demos
        
        # Initialize data holder
        self.latest_data = {
            'cropped_pointcloud': None
        }
        
        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'
        
        # Topics
        cropped_pcd_topic = '/segmented_pointcloud'
        
        # Subscribers
        self.pcd_sub = self.create_subscription(PointCloud2, cropped_pcd_topic, self.pcd_callback, 10)
        
        # Create timer for model rollout
        self.timer = self.create_timer(0.033, self.visualization_callback)  # ~30 FPS
        
        self.get_logger().info("DeploymentNode started and subscribed to topics.")

    def get_tool0_pose(self):
        """Get the current pose of tool0 relative to base_link."""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            translation = trans.transform.translation
            rotation = trans.transform.rotation

            # Create transformation matrix
            T = np.eye(4)
            T[:3, 3] = [translation.x, translation.y, translation.z]
            T[:3, :3] = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()

            return T
        except Exception as e:
            self.get_logger().warn(f'Could not get tool0 pose: {str(e)}')
            return None

    def pcd_callback(self, msg):
        """Callback for the cropped point cloud topic."""
        try:
            # Convert PointCloud2 message to numpy array
            points = np.array([[x, y, z] for x, y, z in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)])
            self.latest_data['cropped_pointcloud'] = points
            self.get_logger().debug(f"Received point cloud with {len(points)} points")
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def visualization_callback(self):
        if self.latest_data['cropped_pointcloud'] is None:
            return

        try:
            # Get tool0 pose
            T_w_e = self.get_tool0_pose()
            if T_w_e is None:
                return

            # Get cropped point cloud
            pcd_w = self.latest_data['cropped_pointcloud']
            
            # Debug: Print demo information
            self.get_logger().info(f"Number of demos: {len(self.demos)}")
            if len(self.demos) > 0:
                self.get_logger().info(f"First demo keys: {self.demos[0].keys()}")
                if 'obs' in self.demos[0]:
                    self.get_logger().info(f"First demo obs length: {len(self.demos[0]['obs'])}")
            
            # Prepare data for model rollout
            full_sample = {
                'demos': self.demos,  # Use the loaded demos
                'live': {
                    'obs': [],
                    'grips': [],
                    'actions_grip': [],
                    'T_w_es': [],
                    'actions': []
                }
            }
            
            # Debug: Print full_sample structure
            self.get_logger().info(f"full_sample demos length: {len(full_sample['demos'])}")
            if len(full_sample['demos']) > 0:
                self.get_logger().info(f"full_sample first demo keys: {full_sample['demos'][0].keys()}")
            
            # Set live data
            full_sample['live']['obs'].append(transform_pcd(subsample_pcd(pcd_w), np.linalg.inv(T_w_e)))
            full_sample['live']['grips'].append(0.0)  # Assuming gripper is open
            full_sample['live']['actions_grip'].append(np.zeros(8))
            full_sample['live']['T_w_es'].append(T_w_e)
            full_sample['live']['actions'].append(T_w_e.reshape(1, 4, 4).repeat(self.model.config['pre_horizon'], axis=0))
            
            # Convert to model input format
            data = save_sample(full_sample, None)
            
            # For efficiency, pre-compute and cache geometry embeddings for the demos
            if not hasattr(self, 'demo_scene_node_embds'):
                self.demo_scene_node_embds, self.demo_scene_node_pos = self.model.model.get_demo_scene_emb(
                    data.to(self.model.config['device']))
            
            # Get live scene embeddings
            data.live_scene_node_embds, data.live_scene_node_pos =\
                self.model.model.get_live_scene_emb(data.to(self.model.config['device']))
            data.demo_scene_node_embds = self.demo_scene_node_embds.clone()
            data.demo_scene_node_pos = self.demo_scene_node_pos.clone()
            
            # Inference on the model
            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=self.model.config['device']):
                    actions, grips = self.model.test_step(data.to(self.model.config['device']), 0)
                actions = actions.squeeze().cpu().numpy()
                grips = grips.squeeze().cpu().numpy()
            
            # TODO: Execute the predicted actions using your robot controller
            print("Predicted actions:", actions.shape)
            print(actions)
            print("Predicted grips:", grips.shape)

        except Exception as e:
            self.get_logger().error(f"Error in visualization: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def load_demo_from_folder(folder_path):
    """Load demonstration data from a folder containing cropped PCD files and transformation matrices."""
    # Load transformation matrices
    transform_file = os.path.join(folder_path, 'tool0_transforms.pkl')
    with open(transform_file, 'rb') as f:
        transforms = pickle.load(f)
    
    # Load cropped PCD files
    pcd_dir = os.path.join(folder_path, 'cropped_pcds')
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    
    # Read PCD files
    pcds = []
    for pcd_file in pcd_files:
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
            # Skip header
            points = []
            for line in lines[11:]:  # Skip PCD header
                x, y, z = map(float, line.strip().split())
                points.append([x, y, z])
            pcds.append(np.array(points))
    
    # Create demo dictionary
    demo = {
        'pcds': pcds,
        'T_w_es': transforms['matrices'],
        'grips': np.zeros(len(pcds))  # Assuming gripper state is not recorded
    }
    
    # Debug: Print demo information
    print(f"Loaded demo from {folder_path}:")
    print(f"  Number of PCDs: {len(pcds)}")
    print(f"  Number of transforms: {len(transforms['matrices'])}")
    print(f"  Demo keys: {demo.keys()}")
    
    return demo

def main(args=None):
    rclpy.init(args=args)
    
    ####################################################################################################################
    # Define rollout parameters. 
    num_demos = 1 # originally 2
    num_traj_wp = 10
    num_diffusion_iters = 4
    compile_models = False
    max_execution_steps = 100
    ####################################################################################################################
    # Load and prepare trained model.
    checkpoint_default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model_path = './checkpoints'
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
    config['num_layers'] = 2

    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = num_demos
    config['num_diffusion_iters_test'] = num_diffusion_iters

    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=False,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=max(num_demos, 1))
    model.eval()

    if compile_models:
        model.model.compile_models()
    ####################################################################################################################
    # Process demonstrations.
    # Load demonstrations from assets/rs_data folder
    demo_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'rs_data')
    
    # Get all demo folders and sort by timestamp (folder name) in descending order
    demo_folders = glob.glob(os.path.join(demo_base_dir, '*'))
    demo_folders.sort(key=lambda x: os.path.basename(x), reverse=True)  # Sort by folder name (timestamp) in descending order
    
    # Take the most recent num_demos folders
    demo_folders = demo_folders[:num_demos]
    
    print(f"Loading {len(demo_folders)} most recent demonstrations:")
    for folder in demo_folders:
        print(f"  - {os.path.basename(folder)}")
    
    demos = []    
    for folder in demo_folders:
        demo = load_demo_from_folder(folder)
        demos.append(demo)
    
    # Create and run the deployment node
    node = DeploymentNode(model, num_demos, num_traj_wp, num_diffusion_iters, demos)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
