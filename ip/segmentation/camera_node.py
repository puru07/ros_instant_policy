#!/usr/bin/env python3

"""
ROS2 Camera Node for RGB-D Data Collection

This script implements a ROS2 node that subscribes to RGB and depth data from a RealSense camera
and saves synchronized RGB images and depth images. The node:

1. Subscribes to the following topics:
   - /camera/camera/color/image_raw (RGB images)
   - /camera/camera/aligned_depth_to_color/image_raw (Aligned depth images)
   - /camera/camera/color/camera_info (Camera parameters)

2. Saves data at regular intervals (default: 2 Hz):
   - RGB images as PNG files
   - Depth images as PNG files (16-bit)
   - All files are saved with timestamps in their names

3. Creates a timestamped directory structure:
   - Base directory: <package_path>/assets/rs_data/
   - Subdirectory: YYYYMMDD_HHMMSS/
   - Files are saved as:
     * rgb_YYYYMMDD_HHMMSS_ffffff.png
     * depth_YYYYMMDD_HHMMSS_ffffff.png

Usage:
    ros2 run ros_instant_policy camera_node

Note: Make sure the RealSense camera is properly connected and the ROS2 drivers are running.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime
import os

class CameraDataHolder:
    def __init__(self):
        self.rgb_image = None
        self.depth_image = None
        self.camera_info = None
        self.last_save_time = 0.0

class CameraSubscriberNode(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')

        # Initialize data holder and bridge
        self.latest_data = CameraDataHolder()
        self.bridge = CvBridge()

        # Create timestamped save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'assets', 'rs_data', timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"Created save directory: {self.save_dir}")

        # Topics
        rgb_topic = '/camera/camera/color/image_raw'
        depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        color_info_topic = '/camera/camera/color/camera_info'

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.color_info_sub = self.create_subscription(CameraInfo, color_info_topic, self.color_info_callback, 10)

        # Create timer for saving data
        self.timer = self.create_timer(0.5, self.save_data_callback)  # 0.5 seconds = 2 Hz

        self.get_logger().info("CameraSubscriberNode started and subscribed to topics.")

    def rgb_callback(self, msg):
        self.latest_data.rgb_image = msg
        self.get_logger().debug("Received new RGB image.")

    def depth_callback(self, msg):
        self.latest_data.depth_image = msg
        self.get_logger().debug("Received new depth image.")

    def color_info_callback(self, msg):
        self.latest_data.camera_info = msg
        self.get_logger().debug("Received new CameraInfo.")

    def save_data_callback(self):
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        
        # Check if we have both RGB and depth data
        if self.latest_data.rgb_image is None or self.latest_data.depth_image is None:
            return

        # Check if enough time has passed since last save
        if current_time - self.latest_data.last_save_time < 0.2:
            return

        try:
            # Convert ROS Image messages to OpenCV format
            rgb_cv = self.bridge.imgmsg_to_cv2(self.latest_data.rgb_image, "bgr8")
            depth_cv = self.bridge.imgmsg_to_cv2(self.latest_data.depth_image, "passthrough")

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            
            # Save RGB image
            rgb_filename = os.path.join(self.save_dir, f'rgb_{timestamp}.png')
            cv2.imwrite(rgb_filename, rgb_cv)
            
            # Save depth image (16-bit)
            depth_filename = os.path.join(self.save_dir, f'depth_{timestamp}.png')
            cv2.imwrite(depth_filename, depth_cv)

            self.get_logger().info(f"Saved RGB image to {rgb_filename}")
            self.get_logger().info(f"Saved depth image to {depth_filename}")
            
            # Update last save time
            self.latest_data.last_save_time = current_time

        except Exception as e:
            self.get_logger().error(f"Error saving data: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
