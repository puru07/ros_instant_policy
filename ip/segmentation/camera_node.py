import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
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

        # Create save directory
        self.save_dir = os.path.join(os.path.expanduser('~'), 'camera_data')
        os.makedirs(self.save_dir, exist_ok=True)

        # Topics - replace these with your actual topic names
        rgb_topic = '/camera/camera/color/image_raw'
        color_info_topic = '/camera/camera/color/camera_info'
        depth_topic = '/camera/camera/depth/image_raw'
        depth_info_topic = '/camera/camera/depth/camera_info'

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.color_info_sub = self.create_subscription(CameraInfo, color_info_topic, self.color_info_callback, 10)
        self.depth_info_sub = self.create_subscription(CameraInfo, depth_info_topic, self.depth_info_callback, 10)

        # Create timer for saving data
        self.timer = self.create_timer(0.2, self.save_data_callback)  # 0.2 seconds = 5 Hz

        self.get_logger().info("CameraSubscriberNode started and subscribed to topics.")

    def rgb_callback(self, msg):
        self.latest_data.rgb_image = msg
        self.get_logger().debug("Received new RGB image.")

    def depth_callback(self, msg):
        self.latest_data.depth_image = msg
        self.get_logger().debug("Received new Depth image.")

    def color_info_callback(self, msg):
        self.latest_data.camera_info = msg
        self.get_logger().debug("Received new CameraInfo.")

    def depth_info_callback(self, msg):
        self.latest_data.camera_info = msg
        self.get_logger().debug("Received new CameraInfo.")

    def save_data_callback(self):
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        
        # Check if we have both RGB and depth images
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
            
            # Convert depth image to point cloud and save
            if self.latest_data.camera_info is not None:
                # Create point cloud from depth image
                height, width = depth_cv.shape
                fx = self.latest_data.camera_info.k[0]
                fy = self.latest_data.camera_info.k[4]
                cx = self.latest_data.camera_info.k[2]
                cy = self.latest_data.camera_info.k[5]

                # Create mesh grid of pixel coordinates
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                
                # Convert to 3D points
                z = depth_cv.astype(np.float32) / 1000.0  # Convert to meters
                x = (x - cx) * z / fx
                y = (y - cy) * z / fy

                # Stack coordinates and reshape
                points = np.stack([x, y, z], axis=-1)
                points = points.reshape(-1, 3)

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # Save point cloud
                pcd_filename = os.path.join(self.save_dir, f'depth_{timestamp}.pcd')
                o3d.io.write_point_cloud(pcd_filename, pcd)

                self.get_logger().info(f"Saved RGB image to {rgb_filename}")
                self.get_logger().info(f"Saved depth point cloud to {pcd_filename}")
                
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
