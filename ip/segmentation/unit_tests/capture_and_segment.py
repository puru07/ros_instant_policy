import sys
sys.path.insert(0, '/home/mcqueen/anaconda3/envs/ip_env/lib/python3.10/site-packages')  # Adjust this path

import torch
import cv2
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor
import os
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize data holder
        self.latest_image = None
        self.image_received = False
        
        # Create subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.get_logger().info('Image capture node initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received = True
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')

def setup_model(model_type="vit_t", checkpoint_path="./checkpoints/sam/mobile_sam.pt"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found at {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    return mobile_sam, device

def process_image(image, predictor, max_size=512):
    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    if scale < 1:  # Only resize if image is larger than max_size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h}")

    # Convert to RGB for SAM
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image_rgb)
    
    # Get center point
    h, w = image_rgb.shape[:2]
    point_coords = np.array([[w//2, h//2]])
    point_labels = np.array([1])  # 1 indicates foreground point
    
    # Generate masks
    start_time = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    
    # Filter masks by score
    if masks is not None:
        mask_score_pairs = [(mask, score) for mask, score in zip(masks, scores) if score > 0.98]
        if mask_score_pairs:
            filtered_masks, filtered_scores = zip(*mask_score_pairs)
            print(f"Found {len(masks)} total masks, {len(filtered_masks)} masks with score > 0.98")
            return image, filtered_masks, filtered_scores
    return image, None, None

def save_visualization(image, masks, scores, output_dir, timestamp):
    if masks is None:
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    original_path = os.path.join(output_dir, f'original_{timestamp}.png')
    cv2.imwrite(original_path, image)
    
    # Save segmented images
    for i, (mask, score) in enumerate(zip(masks, scores)):
        visualization = image.copy()
        overlay = np.zeros_like(visualization)
        overlay[mask] = [255, 0, 0]  # Red color for mask
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
        
        # Draw center point
        h, w = image.shape[:2]
        cv2.circle(visualization, (w//2, h//2), 5, (0, 255, 0), -1)  # Green dot for center point
        
        output_path = os.path.join(output_dir, f'segmented_{timestamp}_mask_{i}.png')
        cv2.imwrite(output_path, visualization)
        print(f"Saved mask {i} visualization to: {output_path}")
        print(f"Score: {score:.3f}")
        print(f"Area: {np.sum(mask)} pixels")

def main():
    # Initialize ROS
    rclpy.init()
    
    # Create node
    node = ImageCaptureNode()
    
    # Setup model
    mobile_sam, device = setup_model()
    
    # Initialize predictor
    predictor = SamPredictor(mobile_sam)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('assets', f'capture_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Capture and process 10 images
        for i in range(10):
            print(f"\nWaiting for image {i+1}/10...")
            
            # Wait for new image
            while not node.image_received:
                rclpy.spin_once(node, timeout_sec=0.1)
            
            # Process image
            processed_image, masks, scores = process_image(node.latest_image, predictor)
            
            # Save results
            image_timestamp = f"{timestamp}_{i:02d}"
            save_visualization(processed_image, masks, scores, output_dir, image_timestamp)
            
            # Reset flag
            node.image_received = False
            
            # Wait for 1 second
            time.sleep(1)
            
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        print(f"\nAll images saved to: {output_dir}")

if __name__ == "__main__":
    main() 