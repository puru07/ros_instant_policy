import torch
import cv2
import numpy as np
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import argparse
import time

def setup_model(model_type="vit_t", checkpoint_path="../checkpoints/sam/mobile_sam.pt"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found at {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    return mobile_sam, device

def load_image(image_path, max_size=512):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Invalid image dimensions")

    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    if scale < 1:  # Only resize if image is larger than max_size
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def point_based_segmentation(predictor, image, point_coords=None):
    h, w = image.shape[:2]
    
    # If no point is provided, use center of image
    if point_coords is None:
        point_coords = np.array([[w//2, h//2]])
    
    point_labels = np.array([1])  # 1 indicates foreground point

    try:
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        return masks, scores
    except Exception as e:
        print(f"Error during point-based mask prediction: {str(e)}")
        return None, None

def automatic_segmentation(mask_generator, image):
    try:
        start_time = time.time()
        masks = mask_generator.generate(image)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        
        # Filter masks by score
        if masks is not None:
            filtered_masks = [mask for mask in masks if mask.get('predicted_iou', 0) > 0.98]
            print(f"Found {len(masks)} total masks, {len(filtered_masks)} masks with score > 0.98")
            return filtered_masks
        return None
    except Exception as e:
        print(f"Error during automatic mask generation: {str(e)}")
        return None

def save_visualization(image, masks, scores=None, output_prefix='../assets/masks'):
    if masks is None:
        return

    if isinstance(masks, list):  # Automatic segmentation masks
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            score = mask_data.get('predicted_iou', 0)
            
            visualization = image.copy()
            overlay = np.zeros_like(visualization)
            overlay[mask] = [255, 0, 0]  # Red color for mask
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
            
            # Convert back to BGR for saving
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
            output_path = f'{output_prefix}_auto_{i}.png'
            cv2.imwrite(output_path, visualization)
            print(f"Saved mask {i} visualization to: {output_path}")
            print(f"Score: {score:.3f}")
            print(f"Area: {np.sum(mask)} pixels")
    else:  # Point-based segmentation masks
        for i, (mask, score) in enumerate(zip(masks, scores)):
            visualization = image.copy()
            overlay = np.zeros_like(visualization)
            overlay[mask] = [255, 0, 0]  # Red color for mask
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
            
            # Convert back to BGR for saving
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
            output_path = f'{output_prefix}_point_{i}.png'
            cv2.imwrite(output_path, visualization)
            print(f"Saved mask {i} visualization to: {output_path}")
            print(f"Score: {score:.3f}")
            print(f"Area: {np.sum(mask)} pixels")

def main():
    parser = argparse.ArgumentParser(description='Mobile SAM Image Segmentation')
    parser.add_argument('--mode', type=str, choices=['point', 'auto'], default='point',
                      help='Segmentation mode: point-based or automatic')
    parser.add_argument('--image', type=str, default='../assets/logo2.png',
                      help='Path to input image')
    parser.add_argument('--model', type=str, default='../checkpoints/sam/mobile_sam.pt',
                      help='Path to model checkpoint')
    parser.add_argument('--point', type=int, nargs=2, default=None,
                      help='Point coordinates for point-based segmentation (x y)')
    parser.add_argument('--max_size', type=int, default=512,
                      help='Maximum dimension for image resizing (default: 512)')
    parser.add_argument('--score_threshold', type=float, default=0.98,
                      help='Minimum score threshold for masks (default: 0.98)')
    args = parser.parse_args()

    # Setup model
    mobile_sam, device = setup_model(checkpoint_path=args.model)
    
    # Load and resize image
    image = load_image(args.image, max_size=args.max_size)
    h, w = image.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    if args.mode == 'point':
        predictor = SamPredictor(mobile_sam)
        predictor.set_image(image)
        
        point_coords = None
        if args.point:
            point_coords = np.array([args.point])
        
        start_time = time.time()
        masks, scores = point_based_segmentation(predictor, image, point_coords)
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        
        if masks is not None:
            # Filter masks by score
            mask_score_pairs = [(mask, score) for mask, score in zip(masks, scores) if score > args.score_threshold]
            if mask_score_pairs:
                filtered_masks, filtered_scores = zip(*mask_score_pairs)
                print(f"Found {len(masks)} total masks, {len(filtered_masks)} masks with score > {args.score_threshold}")
                save_visualization(image, filtered_masks, filtered_scores)
            else:
                print(f"No masks found with score > {args.score_threshold}")
    else:  # auto mode
        mask_generator = SamAutomaticMaskGenerator(
            mobile_sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.86,
            box_nms_thresh=0.7,
        )
        masks = automatic_segmentation(mask_generator, image)
        if masks is not None:
            save_visualization(image, masks)

if __name__ == "__main__":
    main()