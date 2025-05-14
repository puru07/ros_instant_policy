#!/usr/bin/env python3

"""
Tool0 Transform Saver Script

This script subscribes to the transform between base_link and tool0 frames,
converts them to numpy arrays, and saves them in a pickle file.

The saved data includes:
1. Timestamps
2. Transformation matrices (4x4 numpy arrays)
3. Translations and rotations separately

Usage:
    ros2 run ip save_tool0_transforms
"""

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import numpy as np
from datetime import datetime
import os
import pickle
from scipy.spatial.transform import Rotation as R

class Tool0TransformSaver(Node):
    def __init__(self):
        super().__init__('tool0_transform_saver')
        
        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Frame names
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'
        
        # Initialize data storage
        self.transforms = {
            'timestamps': [],
            'matrices': [],
            'translations': [],
            'rotations': []
        }
        
        # Create save directory
        self.setup_save_directory()
        
        # Create timer for transform collection
        self.timer = self.create_timer(0.1, self.collect_transform)  # 10 Hz
        
        self.get_logger().info("Tool0 Transform Saver started")

    def setup_save_directory(self):
        """Create directory for saving transform data."""
        # Create base directory if it doesn't exist
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'rs_data')
        os.makedirs(base_dir, exist_ok=True)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(base_dir, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create transform file path
        self.transform_file = os.path.join(self.save_dir, 'tool0_transforms.pkl')
        
        self.get_logger().info(f"Created save directory: {self.save_dir}")

    def collect_transform(self):
        """Collect the current transform and save it."""
        try:
            # Get latest transform
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time()
            )
            
            # Get timestamp
            timestamp = trans.header.stamp.sec + trans.header.stamp.nanosec * 1e-9
            
            # Get translation and rotation
            translation = trans.transform.translation
            rotation = trans.transform.rotation
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, 3] = [translation.x, translation.y, translation.z]
            T[:3, :3] = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
            
            # Print transform data
            print("\nTransform data being saved:")
            print("Timestamp:", timestamp)
            print("\nTransform Matrix:")
            print(T)
            print("Matrix type:", type(T))
            print("Matrix shape:", T.shape)
            print("\nTranslation:", [translation.x, translation.y, translation.z])
            print("Translation type:", type([translation.x, translation.y, translation.z]))
            print("\nRotation (quaternion):", [rotation.x, rotation.y, rotation.z, rotation.w])
            print("Rotation type:", type([rotation.x, rotation.y, rotation.z, rotation.w]))
            
            # Store data
            self.transforms['timestamps'].append(timestamp)
            self.transforms['matrices'].append(T)
            self.transforms['translations'].append([translation.x, translation.y, translation.z])
            self.transforms['rotations'].append([rotation.x, rotation.y, rotation.z, rotation.w])
            
            # Save data periodically (every 100 transforms)
            if len(self.transforms['timestamps']) % 100 == 0:
                self.save_transforms()
                self.get_logger().info(f"Saved {len(self.transforms['timestamps'])} transforms")
            
        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {str(e)}')

    def save_transforms(self):
        """Save collected transforms to pickle file."""
        with open(self.transform_file, 'wb') as f:
            pickle.dump(self.transforms, f)

    def cleanup(self):
        """Save final transforms and cleanup."""
        self.save_transforms()
        self.get_logger().info(f"Final save complete. Total transforms: {len(self.transforms['timestamps'])}")

def main(args=None):
    rclpy.init(args=args)
    node = Tool0TransformSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cleanup()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 