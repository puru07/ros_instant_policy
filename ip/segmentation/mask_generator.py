#!/usr/bin/env python3

"""
Mask Generator Script for Object Tracking and Segmentation

This script implements an interactive object tracking and segmentation system using MobileSAM (Segment Anything Model).
It allows users to:
1. Select an object of interest by clicking on it in the first frame
2. Use the center of the previous mask as the prompt point for the next frame
3. Generate segmentation masks for each frame
4. Create visualization images showing the tracked object and segmentation
5. Convert depth images to PCD files

The script processes a sequence of images and produces:
- Binary masks in the 'masks/' subdirectory
- Visualization images in the 'output_images/' subdirectory showing:
  * The original image
  * The segmented area (semi-transparent red overlay)
  * The tracked point (green circle)
- PCD files in the 'pcds/' subdirectory

Usage:
    python3 mask_generator.py --image_folder <path_to_images> --checkpoint <path_to_mobile_sam_checkpoint>

Example:
    python3 mask_generator.py --image_folder ../assets/rs_data/test_images_1 --checkpoint ../checkpoints/sam/mobile_sam.pt
"""

import os
import glob
import numpy as np
import cv2
import torch
from mobile_sam import sam_model_registry, SamPredictor
from datetime import datetime

class PointSelector:
    def __init__(self):
        self.point = None
        self.window_name = "Select Point - Press 'q' to confirm"

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
            if key == ord('q') and self.point is not None:
                break
        
        cv2.destroyWindow(self.window_name)
        return self.point

def load_images_sorted(folder_path):
    rgb_paths = sorted(glob.glob(os.path.join(folder_path, '*rgb*.png')))
    depth_paths = sorted(glob.glob(os.path.join(folder_path, '*depth*.png')))
    return rgb_paths, depth_paths

def setup_model(model_type="vit_t", checkpoint_path="./checkpoints/sam/mobile_sam.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)
    return predictor, device

def get_mask_center(mask):
    """Calculate the center of mass of the binary mask."""
    # Find all non-zero points in the mask
    y_coords, x_coords = np.nonzero(mask)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    # Calculate the center of mass
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    return np.array([[center_x, center_y]], dtype=np.float32)

def create_visualization(image, mask, point):
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

def depth_to_pointcloud(depth_img, mask=None, fx=525.0, fy=525.0, cx=319.5, cy=239.5):
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
    x = ((c[valid] - cx) * z / fx).flatten()
    y = ((r[valid] - cy) * z / fy).flatten()
    
    # Stack coordinates and remove any points with invalid values
    points = np.stack([x, y, z], axis=1)
    valid_points = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
    points = points[valid_points]
    
    return points

def write_pcd_file(points, filename):
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

def run_tracking(image_folder, checkpoint_path="./checkpoints/sam/mobile_sam.pt"):
    # Load model
    predictor, device = setup_model(checkpoint_path=checkpoint_path)

    # Load images
    rgb_paths, depth_paths = load_images_sorted(image_folder)
    print(f"Found {len(rgb_paths)} RGB images and {len(depth_paths)} depth images.")

    if len(rgb_paths) < 2:
        print("Need at least 2 RGB images.")
        return

    # Create output directories
    base_dir = os.path.dirname(rgb_paths[0])
    masks_dir = os.path.join(base_dir, 'masks')
    output_dir = os.path.join(base_dir, 'output_images')
    pcd_dir = os.path.join(base_dir, 'pcds')
    cropped_pcd_dir = os.path.join(base_dir, 'cropped_pcds')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(cropped_pcd_dir, exist_ok=True)
    print(f"Created output directories: {masks_dir}, {output_dir}, {pcd_dir}, and {cropped_pcd_dir}")

    # Load first image
    prev_img = cv2.imread(rgb_paths[0])
    if prev_img is None:
        print(f"Failed to load image: {rgb_paths[0]}")
        return

    # Let user select point
    point_selector = PointSelector()
    print("Click on the object you want to track in the first frame, then press 'q' to confirm.")
    prev_point = point_selector.select_point(prev_img)
    
    if prev_point is None:
        print("No point selected. Exiting.")
        return

    # Convert to RGB for SAM
    prev_img_rgb = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
    predictor.set_image(prev_img_rgb)

    # Segment the first frame
    point_labels = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=prev_point.reshape(-1, 2),
        point_labels=point_labels,
        multimask_output=False
    )
    mask = masks[0]
    
    # Save mask
    mask_filename = os.path.join(masks_dir, os.path.basename(rgb_paths[0]).replace('.png', '_mask.png'))
    cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
    print(f"Saved first mask: {mask_filename}")
    
    # Save visualization
    vis_img = create_visualization(prev_img, mask, prev_point)
    vis_filename = os.path.join(output_dir, os.path.basename(rgb_paths[0]).replace('.png', '_vis.png'))
    cv2.imwrite(vis_filename, vis_img)
    print(f"Saved first visualization: {vis_filename}")

    # Process remaining frames
    for idx in range(1, len(rgb_paths)):
        curr_img = cv2.imread(rgb_paths[idx])
        if curr_img is None:
            print(f"Failed to load image: {rgb_paths[idx]}")
            continue

        # Get the center of the previous mask as the prompt point
        next_point = get_mask_center(mask)
        if next_point is None:
            print(f"Could not find center of mask for frame {idx}. Skipping.")
            continue
        
        # Convert to RGB for SAM
        curr_img_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
        predictor.set_image(curr_img_rgb)
        
        # Segment current frame
        masks, scores, logits = predictor.predict(
            point_coords=next_point.reshape(-1, 2),
            point_labels=point_labels,
            multimask_output=False
        )
        mask = masks[0]
        
        # Save mask
        mask_filename = os.path.join(masks_dir, os.path.basename(rgb_paths[idx]).replace('.png', '_mask.png'))
        cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
        print(f"Saved mask for frame {idx}: {mask_filename}")
        
        # Save visualization
        vis_img = create_visualization(curr_img, mask, next_point)
        vis_filename = os.path.join(output_dir, os.path.basename(rgb_paths[idx]).replace('.png', '_vis.png'))
        cv2.imwrite(vis_filename, vis_img)
        print(f"Saved visualization for frame {idx}: {vis_filename}")

        # Process corresponding depth image
        if idx < len(depth_paths):
            try:
                # Read depth image
                depth_img = cv2.imread(depth_paths[idx], cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    print(f"Failed to load depth image: {depth_paths[idx]}")
                    continue

                # Convert depth image to full point cloud
                points = depth_to_pointcloud(depth_img)
                
                # Save full point cloud
                pcd_filename = os.path.join(pcd_dir, os.path.basename(depth_paths[idx]).replace('.png', '.pcd'))
                write_pcd_file(points, pcd_filename)
                print(f"Saved full point cloud for frame {idx}: {pcd_filename}")

                # Convert depth image to cropped point cloud using mask
                cropped_points = depth_to_pointcloud(depth_img, mask)
                
                # Save cropped point cloud
                cropped_pcd_filename = os.path.join(cropped_pcd_dir, os.path.basename(depth_paths[idx]).replace('.png', '_cropped.pcd'))
                write_pcd_file(cropped_points, cropped_pcd_filename)
                print(f"Saved cropped point cloud for frame {idx}: {cropped_pcd_filename}")

            except Exception as e:
                print(f"Error processing depth image for frame {idx}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Track and segment objects in image sequence')
    parser.add_argument('--image_folder', type=str, default="../assets/rs_data/test_images_3", help='Folder containing image sequence')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/sam/mobile_sam.pt", 
                      help='Path to MobileSAM checkpoint')
    args = parser.parse_args()
    
    run_tracking(args.image_folder, args.checkpoint)
