import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import json

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        
        # Draw bounding box
        bbox = ann['bbox']
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img)

def detect_circular_objects(image, mask_segmentation, min_radius=5, max_radius=100, 
                          param1=50, param2=30, min_dist=20):
    """
    Detect circular objects within a specific mask region.
    
    Args:
        image: Original image (numpy array)
        mask_segmentation: Boolean mask for the specific region
        min_radius: Minimum circle radius to detect
        max_radius: Maximum circle radius to detect
        param1: First method-specific parameter for HoughCircles
        param2: Accumulator threshold for HoughCircles
        min_dist: Minimum distance between circle centers
    
    Returns:
        circles: Detected circles as (x, y, radius) tuples
        masked_image: Image with only the masked region visible
    """
    # Create a masked image (only show the region of interest)
    masked_image = np.zeros_like(image)
    masked_image[mask_segmentation] = image[mask_segmentation]
    
    # Convert to grayscale for circle detection
    gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Verify that the circle center is within the mask
            if mask_segmentation[y, x]:
                detected_circles.append((x, y, r))
    
    return detected_circles, masked_image

def save_crop_boxes_with_circles(image, masks, output_folder="crop_boxes_with_circles"):
    """
    Save crop box images from masks and detect circular objects within each mask.
    
    Args:
        image: Original image (numpy array)
        masks: List of mask dictionaries from SAM
        output_folder: Folder to save crop box images and circle detection results
    """
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    circles_folder = os.path.join(output_folder, "circles_detected")
    os.makedirs(circles_folder, exist_ok=True)
    
    print(f"Analyzing {len(masks)} masks for circular objects...")
    print(f"Original image shape: {image.shape}")
    
    # Store results for all masks
    all_results = []
    
    for i, mask in enumerate(masks):
        bbox = mask['bbox']  # XYWH format
        x, y, w, h = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        x2 = max(0, min(x + w, img_width))
        y2 = max(0, min(y + h, img_height))
        
        # Skip if crop is too small
        if (x2 - x) < 10 or (y2 - y) < 10:
            continue
        
        # Extract crop from original image
        crop_image = image[y:y2, x:x2]
        
        # Get the mask segmentation for this region
        mask_seg = mask['segmentation'][y:y2, x:x2]
        
        # Detect circular objects in this masked region
        circles, masked_crop = detect_circular_objects(crop_image, mask_seg)
        
        # Store results
        mask_result = {
            'mask_id': i,
            'bbox': bbox,
            'area': mask['area'],
            'predicted_iou': mask['predicted_iou'],
            'circles_detected': len(circles),
            'circles': [{'x': int(x), 'y': int(y), 'radius': int(r)} for x, y, r in circles]
        }
        all_results.append(mask_result)
        
        # Save the crop image
        filename = f"crop_{i:04d}_area_{mask['area']}_circles_{len(circles)}.jpg"
        filepath = os.path.join(output_folder, filename)
        crop_bgr = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, crop_bgr)
        
        # If circles were detected, save an annotated version
        if circles:
            annotated_crop = crop_image.copy()
            for (cx, cy, r) in circles:
                # Draw circle and center point
                cv2.circle(annotated_crop, (cx, cy), r, (255, 0, 0), 2)
                cv2.circle(annotated_crop, (cx, cy), 2, (0, 255, 0), 3)
            
            # Save annotated image
            circles_filename = f"circles_{i:04d}_count_{len(circles)}.jpg"
            circles_filepath = os.path.join(circles_folder, circles_filename)
            annotated_bgr = cv2.cvtColor(annotated_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(circles_filepath, annotated_bgr)
            
            print(f"Mask {i}: Found {len(circles)} circular objects")
    
    # Save JSON summary of all results
    summary_file = os.path.join(output_folder, "circular_objects_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary statistics
    total_circles = sum(result['circles_detected'] for result in all_results)
    masks_with_circles = sum(1 for result in all_results if result['circles_detected'] > 0)
    
    print(f"\n=== CIRCULAR OBJECTS DETECTION SUMMARY ===")
    print(f"Total masks analyzed: {len(all_results)}")
    print(f"Masks with circular objects: {masks_with_circles}")
    print(f"Total circular objects detected: {total_circles}")
    print(f"Results saved to: {output_folder}")
    print(f"Detailed results saved to: {summary_file}")
    
    return all_results

def save_crop_boxes(image, masks, output_folder="crop_boxes"):
    """
    Save crop box images from masks to a specified folder.
    
    Args:
        image: Original image (numpy array)
        masks: List of mask dictionaries from SAM
        output_folder: Folder to save crop box images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Saving {len(masks)} crop box images to '{output_folder}' folder...")
    print(f"Original image shape: {image.shape}")
    
    # Debug: check what keys are available and print some examples
    if masks:
        print(f"Available keys in mask: {list(masks[0].keys())}")
        print(f"First few crop_boxes: {[mask['crop_box'] for mask in masks[:3]]}")
        print(f"First few bboxes: {[mask['bbox'] for mask in masks[:3]]}")
    
    for i, mask in enumerate(masks):
        # Use bbox instead of crop_box since crop_box seems to be full image for all
        bbox = mask['bbox']  # XYWH format
        x, y, w, h = map(int, bbox)  # Convert to integers
        
        # Ensure coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        x2 = max(0, min(x + w, img_width))
        y2 = max(0, min(y + h, img_height))
        
        # Extract crop from original image (note: numpy indexing is [y:y2, x:x2])
        crop_image = image[y:y2, x:x2]
        
        # Debug info for first few crops
        if i < 5:
            print(f"Crop {i}: bbox={bbox}, actual_coords=({x},{y},{x2},{y2}), crop_shape={crop_image.shape}")
        
        # Skip if crop is too small
        if crop_image.shape[0] < 5 or crop_image.shape[1] < 5:
            print(f"Skipping crop {i}: too small ({crop_image.shape})")
            continue
        
        # Save the crop as an image file
        filename = f"crop_{i:04d}_area_{mask['area']}_iou_{mask['predicted_iou']:.3f}.jpg"
        filepath = os.path.join(output_folder, filename)
        
        # Convert RGB to BGR for cv2.imwrite
        crop_bgr = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, crop_bgr)
    
    print(f"Successfully saved crop box images!")

def visualize_circles_on_image(image, masks, circular_results, save_path="circular_objects_visualization.jpg"):
    """
    Visualize all detected circular objects on the original image.
    
    Args:
        image: Original image (numpy array)
        masks: List of mask dictionaries from SAM
        circular_results: Results from save_crop_boxes_with_circles function
        save_path: Path to save the visualization image
    """
    # Create a copy of the original image for visualization
    vis_image = image.copy()
    
    # Create a color map for different masks
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    total_circles_drawn = 0
    
    for result in circular_results:
        if result['circles_detected'] > 0:
            mask_id = result['mask_id']
            mask = masks[mask_id]
            bbox = result['bbox']
            x, y, w, h = bbox
            
            # Choose color for this mask
            color = colors[mask_id % len(colors)]
            
            # Draw each circle from this mask
            for circle in result['circles']:
                # Convert local coordinates to global coordinates
                global_x = x + circle['x']
                global_y = y + circle['y']
                radius = circle['radius']
                
                # Draw circle outline
                cv2.circle(vis_image, (int(global_x), int(global_y)), int(radius), color, 2)
                # Draw center point
                cv2.circle(vis_image, (int(global_x), int(global_y)), 3, color, -1)
                
                # Add text label
                label = f"M{mask_id}"
                cv2.putText(vis_image, label, (int(global_x) - 10, int(global_y) - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                total_circles_drawn += 1
    
    # Add summary text to the image
    summary_text = f"Total Circular Objects: {total_circles_drawn}"
    cv2.putText(vis_image, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save the visualization
    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, vis_bgr)
    
    print(f"Visualization saved to: {save_path}")
    print(f"Total circular objects visualized: {total_circles_drawn}")
    
    return vis_image

def analyze_circles_with_filters(image, masks, min_radius=3, max_radius=50, 
                                min_mask_area=100, max_circles_per_mask=10,
                                circularity_threshold=0.7, param1=50, param2=20):
    """
    Advanced circular object detection with filtering options.
    
    Args:
        image: Original image (numpy array)
        masks: List of mask dictionaries from SAM
        min_radius: Minimum circle radius to detect
        max_radius: Maximum circle radius to detect
        min_mask_area: Minimum mask area to consider for analysis
        max_circles_per_mask: Maximum number of circles to detect per mask
        circularity_threshold: Minimum circularity score (0-1) to accept a circle
        param1: HoughCircles param1 (gradient threshold)
        param2: HoughCircles param2 (accumulator threshold)
    
    Returns:
        filtered_results: Results with additional filtering and metrics
    """
    print(f"Analyzing masks with advanced filtering...")
    print(f"Parameters: min_radius={min_radius}, max_radius={max_radius}")
    print(f"Min mask area: {min_mask_area}, Circularity threshold: {circularity_threshold}")
    
    filtered_results = []
    
    for i, mask in enumerate(masks):
        # Skip small masks
        if mask['area'] < min_mask_area:
            continue
            
        bbox = mask['bbox']
        x, y, w, h = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        x2 = max(0, min(x + w, img_width))
        y2 = max(0, min(y + h, img_height))
        
        if (x2 - x) < 10 or (y2 - y) < 10:
            continue
        
        # Extract crop and mask
        crop_image = image[y:y2, x:x2]
        mask_seg = mask['segmentation'][y:y2, x:x2]
        
        # Detect circles with specified parameters
        circles, _ = detect_circular_objects(
            crop_image, mask_seg, 
            min_radius=min_radius, max_radius=max_radius,
            param1=param1, param2=param2, min_dist=max(10, min_radius*2)
        )
        
        # Filter circles by various criteria
        valid_circles = []
        for cx, cy, r in circles:
            # Check if circle is mostly within the mask
            circle_mask = np.zeros_like(mask_seg, dtype=np.uint8)
            cv2.circle(circle_mask, (cx, cy), r, 1, -1)
            
            # Calculate overlap with the original mask
            overlap = np.sum(circle_mask & mask_seg)
            circle_area = np.sum(circle_mask)
            
            if circle_area > 0:
                overlap_ratio = overlap / circle_area
                
                # Calculate circularity (how close to perfect circle)
                contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0
                else:
                    circularity = 0
                
                # Accept circle if it meets criteria
                if (overlap_ratio > 0.6 and circularity > circularity_threshold):
                    valid_circles.append({
                        'x': int(cx), 'y': int(cy), 'radius': int(r),
                        'overlap_ratio': float(overlap_ratio),
                        'circularity': float(circularity),
                        'area': int(circle_area)
                    })
        
        # Limit number of circles per mask
        if len(valid_circles) > max_circles_per_mask:
            # Sort by circularity and take the best ones
            valid_circles.sort(key=lambda c: c['circularity'], reverse=True)
            valid_circles = valid_circles[:max_circles_per_mask]
        
        if valid_circles:
            result = {
                'mask_id': i,
                'bbox': bbox,
                'mask_area': mask['area'],
                'predicted_iou': mask['predicted_iou'],
                'circles_detected': len(valid_circles),
                'circles': valid_circles,
                'avg_circularity': float(np.mean([c['circularity'] for c in valid_circles])),
                'total_circle_area': int(sum(c['area'] for c in valid_circles))
            }
            filtered_results.append(result)
            
            if len(valid_circles) > 0:
                print(f"Mask {i}: {len(valid_circles)} high-quality circular objects "
                      f"(avg circularity: {result['avg_circularity']:.3f})")
    
    return filtered_results

image = cv2.imread('images/ob_in_cell_1.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "../models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

# Save crop box images to folder (original function)
save_crop_boxes(image, masks, "crop_boxes")

# NEW: Analyze each mask for circular objects
circular_results = save_crop_boxes_with_circles(image, masks, "crop_boxes_with_circles")

# NEW: Create visualization of all circular objects on the original image
visualization = visualize_circles_on_image(image, masks, circular_results, "all_circular_objects_detected.jpg")

# NEW: Advanced analysis with filtering
print("\n" + "="*60)
print("RUNNING ADVANCED CIRCULAR OBJECT DETECTION")
print("="*60)

advanced_results = analyze_circles_with_filters(
    image, masks, 
    min_radius=5, max_radius=40,
    min_mask_area=200,
    max_circles_per_mask=5,
    circularity_threshold=0.6,
    param1=50, param2=25
)

# Save advanced results
advanced_output_folder = "advanced_circle_detection"
os.makedirs(advanced_output_folder, exist_ok=True)

# Save JSON summary
with open(os.path.join(advanced_output_folder, "advanced_results.json"), 'w') as f:
    json.dump(advanced_results, f, indent=2)

print(f"\nAdvanced results saved to: {advanced_output_folder}")
print(f"High-quality circular objects found: {sum(r['circles_detected'] for r in advanced_results)}")

# Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
# * `segmentation` : the mask
# * `area` : the area of the mask in pixels
# * `bbox` : the boundary box of the mask in XYWH format
# * `predicted_iou` : the model's own prediction for the quality of the mask
# * `point_coords` : the sampled input point that generated this mask
# * `stability_score` : an additional measure of mask quality
# * `crop_box` : the crop of the image used to generate this mask in XYWH format


