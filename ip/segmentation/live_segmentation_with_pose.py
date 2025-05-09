#!/usr/bin/env python3

"""
Live Object Segmentation with Pose Tracking Script

This script implements real-time object segmentation using MobileSAM (Segment Anything Model)
and tracks the robot's tool0 pose. It allows users to:
1. View live camera feed
2. Select an object of interest by clicking on it
3. Generate segmentation masks in real-time
4. Create visualization showing the tracked object and segmentation
5. Save RGB images, visualizations, cropped depth images, PCD files, and tool0 poses at 5 Hz

Usage:
    ros2 run ip live_segmentation_with_pose --ros-args --checkpoint <path_to_mobile_sam_checkpoint>

Example:
    ros2 run ip live_segmentation_with_pose --ros-args --checkpoint ../checkpoints/sam/mobile_sam.pt
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor
import argparse
from datetime import datetime
import os
import sys
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

class PointSelector:
    def __init__(self):
        self.point = None
        self.window_name = "Live Camera Feed - Click on object to track, press 'q' to quit"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = np.array([[x, y]], dtype=np.float32)
            # Draw a circle at the selected point
            img_copy = self.image.copy()
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, img_copy)

    def select_point(self, image):
        self.image = image.copy()
        self.point = None
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Show image and wait for point selection
        cv2.imshow(self.window_name, self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        return self.point

class LiveSegmentationWithPoseNode(Node):
    def __init__(self, checkpoint_path):
        super().__init__('live_segmentation_with_pose_node')

        # Initialize data holder and bridge
        self.latest_data = {
            'rgb_image': None,
            'depth_image': None,
            'camera_info': None
        }
        self.bridge = CvBridge()

        # Setup MobileSAM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

        # Initialize point selector
        self.point_selector = PointSelector()
        
        # Flag to track if we have selected a point
        self.has_selected_point = False
        self.selected_point = None
        self.current_mask = None

        # Initialize saving variables
        self.save_dir = None
        self.last_save_time = 0.0
        self.save_interval = 0.2  # 5 Hz
        self.first_frame_time = None

        # Initialize TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame = 'base_link'
        self.tool_frame = 'tool0'

        # Camera parameters (default values, will be updated from camera_info)
        self.fx = 525.0
        self.fy = 525.0
        self.cx = 319.5
        self.cy = 239.5

        # Topics
        rgb_topic = '/camera/camera/color/image_raw'
        depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        color_info_topic = '/camera/camera/color/camera_info'

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.color_info_sub = self.create_subscription(CameraInfo, color_info_topic, self.color_info_callback, 10)

        # Create timer for visualization
        self.timer = self.create_timer(0.033, self.visualization_callback)  # ~30 FPS

        self.get_logger().info("LiveSegmentationWithPoseNode started and subscribed to topics.")

    def setup_save_directories(self):
        """Create directories for saving images, PCD files, and poses."""
        if self.save_dir is None:
            # Create base directory if it doesn't exist
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'rs_data')
            os.makedirs(base_dir, exist_ok=True)
            
            # Create timestamped directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_dir = os.path.join(base_dir, timestamp)
            
            # Create subdirectories
            self.rgb_dir = os.path.join(self.save_dir, 'rgb_images')
            self.vis_dir = os.path.join(self.save_dir, 'visualizations')
            self.depth_dir = os.path.join(self.save_dir, 'depth_images')
            self.cropped_depth_dir = os.path.join(self.save_dir, 'cropped_depth_images')
            self.cropped_pcd_dir = os.path.join(self.save_dir, 'cropped_pcds')
            self.pose_dir = os.path.join(self.save_dir, 'poses')
            os.makedirs(self.rgb_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)
            os.makedirs(self.depth_dir, exist_ok=True)
            os.makedirs(self.cropped_depth_dir, exist_ok=True)
            os.makedirs(self.cropped_pcd_dir, exist_ok=True)
            os.makedirs(self.pose_dir, exist_ok=True)
            
            # Create pose file
            self.pose_file = os.path.join(self.pose_dir, 'tool0_poses.txt')
            with open(self.pose_file, 'w') as f:
                f.write("# Tool0 poses relative to base_link\n")
                f.write("# Format: timestamp, x, y, z, qx, qy, qz, qw\n")
            
            self.get_logger().info(f"Created save directories in: {self.save_dir}")

    def get_tool0_pose(self):
        """Get the current pose of tool0 relative to base_link using image timestamp."""
        try:
            if self.latest_data['rgb_image'] is None:
                return None

            stamp = self.latest_data['rgb_image'].header.stamp
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            translation = trans.transform.translation
            rotation = trans.transform.rotation

            return {
                'translation': (translation.x, translation.y, translation.z),
                'rotation': (rotation.x, rotation.y, rotation.z, rotation.w)
            }
        except Exception as e:
            self.get_logger().warn(f'Could not get tool0 pose: {str(e)}')
            return None


    def save_pose(self, timestamp, pose):
        """Save the tool0 pose to file."""
        if pose is None:
            return
        
        with open(self.pose_file, 'a') as f:
            tx, ty, tz = pose['translation']
            qx, qy, qz, qw = pose['rotation']
            f.write(f"{timestamp:.3f}, {tx:.6f}, {ty:.6f}, {tz:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}\n")

    def depth_to_pointcloud(self, depth_img, mask=None):
        """Convert depth image to point cloud using camera intrinsics.
        If mask is provided, only return points within the masked region."""
        # Ensure depth image is 2D
        if len(depth_img.shape) > 2:
            depth_img = depth_img[:, :, 0]  # Take first channel if 3D
        
        rows, cols = depth_img.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Apply mask if provided
        if mask is not None:
            valid = (depth_img > 0) & mask
        else:
            valid = depth_img > 0
        
        # Reshape arrays to match
        z = depth_img[valid].flatten()
        x = ((c[valid] - self.cx) * z / self.fx).flatten()
        y = ((r[valid] - self.cy) * z / self.fy).flatten()
        
        # Stack coordinates and remove any points with invalid values
        points = np.stack([x, y, z], axis=1)
        valid_points = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_points]
        
        return points

    def write_pcd_file(self, points, filename):
        """Write points to a PCD file."""
        with open(filename, 'w') as f:
            # Write PCD header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZES 4 4 4\n")
            f.write("TYPES F F F\n")
            f.write("COUNTS 1 1 1\n")
            f.write(f"WIDTH {len(points)}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {len(points)}\n")
            f.write("DATA ascii\n")
            
            # Write points
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def save_frame(self, rgb_image, vis_image, depth_image, mask):
        """Save RGB, visualization, depth images, cropped PCD files, and tool0 pose if enough time has passed."""
        # Get tool0 pose first - check before any processing
        pose = self.get_tool0_pose()
        if pose is None:
            self.get_logger().warn("Skipping frame save: No tool0 transform available")
            return

        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        
        # Initialize first frame time if not set
        if self.first_frame_time is None:
            self.first_frame_time = current_time
        
        # Check if enough time has passed since last save
        if current_time - self.last_save_time < self.save_interval:
            return
        
        # Generate timestamp for filenames
        frame_time = current_time - self.first_frame_time
        timestamp = f"{frame_time:.3f}"
        
        # Save RGB image
        rgb_filename = os.path.join(self.rgb_dir, f'rgb_{timestamp}.png')
        cv2.imwrite(rgb_filename, rgb_image)
        
        # Save visualization image
        vis_filename = os.path.join(self.vis_dir, f'vis_{timestamp}.png')
        cv2.imwrite(vis_filename, vis_image)
        
        # Save depth image
        depth_filename = os.path.join(self.depth_dir, f'depth_{timestamp}.png')
        cv2.imwrite(depth_filename, depth_image)
        
        # Create and save cropped depth image and its point cloud
        if depth_image is not None and mask is not None:
            # Create a copy of the depth image
            cropped_depth = depth_image.copy()
            # Set pixels outside the mask to 0
            cropped_depth[~mask] = 0
            # Save cropped depth image
            cropped_depth_filename = os.path.join(self.cropped_depth_dir, f'depth_{timestamp}_cropped.png')
            cv2.imwrite(cropped_depth_filename, cropped_depth)
            
            # Convert cropped depth to point cloud
            cropped_points = self.depth_to_pointcloud(depth_image, mask)
            cropped_pcd_filename = os.path.join(self.cropped_pcd_dir, f'depth_{timestamp}_cropped.pcd')
            self.write_pcd_file(cropped_points, cropped_pcd_filename)
            
            # Save tool0 pose
            self.save_pose(float(timestamp), pose)
        
        # Update last save time
        self.last_save_time = current_time
        
        self.get_logger().debug(f"Saved frame at {timestamp}")

    def rgb_callback(self, msg):
        self.latest_data['rgb_image'] = msg

    def depth_callback(self, msg):
        self.latest_data['depth_image'] = msg

    def color_info_callback(self, msg):
        self.latest_data['camera_info'] = msg
        # Update camera parameters from camera info
        self.fx = msg.k[0]  # focal length x
        self.fy = msg.k[4]  # focal length y
        self.cx = msg.k[2]  # principal point x
        self.cy = msg.k[5]  # principal point y

    def get_mask_center(self, mask):
        """Calculate the center of mass of the binary mask."""
        y_coords, x_coords = np.nonzero(mask)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        return np.array([[center_x, center_y]], dtype=np.float32)

    def create_visualization(self, image, mask, point):
        # Create a copy of the image
        vis_img = image.copy()
        
        # Create a colored mask overlay (semi-transparent red)
        mask_overlay = np.zeros_like(vis_img)
        mask_overlay[mask] = [0, 0, 255]  # Red color for mask
        vis_img = cv2.addWeighted(vis_img, 1, mask_overlay, 0.5, 0)
        
        # Draw the prompt point
        if point is not None:
            x, y = point[0].astype(int)
            cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)  # Green circle for point
        
        return vis_img

    def visualization_callback(self):
        if self.latest_data['rgb_image'] is None or self.latest_data['depth_image'] is None:
            return

        try:
            # Convert ROS Image messages to OpenCV format
            rgb_cv = self.bridge.imgmsg_to_cv2(self.latest_data['rgb_image'], "bgr8")
            depth_cv = self.bridge.imgmsg_to_cv2(self.latest_data['depth_image'], "passthrough")
            
            # If we haven't selected a point yet, show the image and wait for selection
            if not self.has_selected_point:
                point = self.point_selector.select_point(rgb_cv)
                if point is not None:
                    self.has_selected_point = True
                    self.selected_point = point
                    # Convert to RGB for SAM
                    rgb_rgb = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(rgb_rgb)
                    # Segment the first frame
                    point_labels = np.array([1])
                    masks, scores, logits = self.predictor.predict(
                        point_coords=self.selected_point.reshape(-1, 2),
                        point_labels=point_labels,
                        multimask_output=False
                    )
                    self.current_mask = masks[0]
                    # Setup save directories when first mask is generated
                    self.setup_save_directories()
            
            # If we have a mask, update it using the center of the previous mask
            elif self.current_mask is not None:
                # Get the center of the previous mask as the prompt point
                next_point = self.get_mask_center(self.current_mask)
                if next_point is not None:
                    # Convert to RGB for SAM
                    rgb_rgb = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
                    self.predictor.set_image(rgb_rgb)
                    # Segment current frame
                    point_labels = np.array([1])
                    masks, scores, logits = self.predictor.predict(
                        point_coords=next_point.reshape(-1, 2),
                        point_labels=point_labels,
                        multimask_output=False
                    )
                    self.current_mask = masks[0]
                    self.selected_point = next_point

            # Create visualization
            if self.current_mask is not None:
                vis_img = self.create_visualization(rgb_cv, self.current_mask, self.selected_point)
                cv2.imshow(self.point_selector.window_name, vis_img)
                # Save frames if we have a mask
                self.save_frame(rgb_cv, vis_img, depth_cv, self.current_mask)
            else:
                cv2.imshow(self.point_selector.window_name, rgb_cv)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in visualization: {str(e)}")

def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)


    # Get checkpoint path from ROS2 parameters
    node = Node('parameter_node')
    checkpoint_path = node.declare_parameter('checkpoint', '../checkpoints/sam/mobile_sam.pt').value
    node.destroy_node()
    
    # Create and run the main node
    node = LiveSegmentationWithPoseNode(checkpoint_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 