#!/usr/bin/env python3

"""
Mask Generator Script for Object Tracking and Segmentation

This script implements an interactive object tracking and segmentation system using MobileSAM (Segment Anything Model).
It allows users to:
1. Select an object of interest by clicking on it in the first frame
2. Use the center of the previous mask as the prompt point for the next frame
3. Generate segmentation masks for each frame
4. Create visualization images showing the tracked object and segmentation
5. Crop point cloud data using the generated masks

The script processes a sequence of images and produces:
- Binary masks in the 'masks/' subdirectory
- Visualization images in the 'output_images/' subdirectory showing:
  * The original image
  * The segmented area (semi-transparent red overlay)
  * The tracked point (green circle)
- Cropped point clouds in the 'cropped_pcds/' subdirectory

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
    image_paths = sorted(
        glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpg'))
    )
    return image_paths

def load_pcd_sorted(folder_path):
    pcd_paths = sorted(glob.glob(os.path.join(folder_path, '*.pcd')))
    return pcd_paths

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

def read_pcd_file(pcd_path):
    """Read points from a PCD file."""
    points = []
    with open(pcd_path, 'r') as f:
        # Skip header
        for _ in range(11):
            next(f)
        # Read points
        for line in f:
            x, y, z = map(float, line.strip().split()[:3])
            points.append([x, y, z])
    return np.array(points)

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

def crop_point_cloud(points, mask, image_shape):
    """Crop point cloud using the mask."""
    # Reshape points to match image dimensions
    h, w = image_shape[:2]
    points_2d = points[:, :2]  # Take only x,y coordinates
    
    # Scale points to image coordinates
    points_2d[:, 0] = (points_2d[:, 0] - points_2d[:, 0].min()) / (points_2d[:, 0].max() - points_2d[:, 0].min()) * w
    points_2d[:, 1] = (points_2d[:, 1] - points_2d[:, 1].min()) / (points_2d[:, 1].max() - points_2d[:, 1].min()) * h
    
    # Convert to integer coordinates
    points_2d = points_2d.astype(int)
    
    # Filter points that are within the mask
    valid_points = []
    for i, (x, y) in enumerate(points_2d):
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            valid_points.append(points[i])
    
    return np.array(valid_points)

def run_tracking(image_folder, checkpoint_path="./checkpoints/sam/mobile_sam.pt"):
    # Load model
    predictor, device = setup_model(checkpoint_path=checkpoint_path)

    # Load images and PCDs
    image_paths = load_images_sorted(image_folder)
    pcd_paths = load_pcd_sorted(image_folder)
    print(f"Found {len(image_paths)} images and {len(pcd_paths)} PCD files.")

    if len(image_paths) < 2 or len(pcd_paths) < 2:
        print("Need at least 2 images and PCD files.")
        return

    # Create output directories
    base_dir = os.path.dirname(image_paths[0])
    masks_dir = os.path.join(base_dir, 'masks')
    output_dir = os.path.join(base_dir, 'output_images')
    pcd_dir = os.path.join(base_dir, 'cropped_pcds')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    print(f"Created output directories: {masks_dir}, {output_dir}, and {pcd_dir}")

    # Load first image
    prev_img = cv2.imread(image_paths[0])
    if prev_img is None:
        print(f"Failed to load image: {image_paths[0]}")
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
    mask_filename = os.path.join(masks_dir, os.path.basename(image_paths[0]).replace('.png', '_mask.png'))
    cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
    print(f"Saved first mask: {mask_filename}")
    
    # Save visualization
    vis_img = create_visualization(prev_img, mask, prev_point)
    vis_filename = os.path.join(output_dir, os.path.basename(image_paths[0]).replace('.png', '_vis.png'))
    cv2.imwrite(vis_filename, vis_img)
    print(f"Saved first visualization: {vis_filename}")

    # Process remaining frames
    for idx in range(1, len(image_paths)):
        curr_img = cv2.imread(image_paths[idx])
        if curr_img is None:
            print(f"Failed to load image: {image_paths[idx]}")
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
        mask_filename = os.path.join(masks_dir, os.path.basename(image_paths[idx]).replace('.png', '_mask.png'))
        cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
        print(f"Saved mask for frame {idx}: {mask_filename}")
        
        # Save visualization
        vis_img = create_visualization(curr_img, mask, next_point)
        vis_filename = os.path.join(output_dir, os.path.basename(image_paths[idx]).replace('.png', '_vis.png'))
        cv2.imwrite(vis_filename, vis_img)
        print(f"Saved visualization for frame {idx}: {vis_filename}")

        # Process corresponding PCD file
        if idx < len(pcd_paths):
            try:
                # Read PCD file
                points = read_pcd_file(pcd_paths[idx])
                
                # Crop point cloud using the mask
                cropped_points = crop_point_cloud(points, mask, curr_img.shape)
                
                # Save cropped point cloud
                pcd_filename = os.path.join(pcd_dir, os.path.basename(pcd_paths[idx]).replace('.pcd', '_cropped.pcd'))
                write_pcd_file(cropped_points, pcd_filename)
                print(f"Saved cropped point cloud for frame {idx}: {pcd_filename}")
            except Exception as e:
                print(f"Error processing PCD file for frame {idx}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Track and segment objects in image sequence')
    parser.add_argument('--image_folder', type=str, default="../assets/rs_data/test_images_2", help='Folder containing image sequence')
    parser.add_argument('--checkpoint', type=str, default="../checkpoints/sam/mobile_sam.pt", 
                      help='Path to MobileSAM checkpoint')
    args = parser.parse_args()
    
    run_tracking(args.image_folder, args.checkpoint)
