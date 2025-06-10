#!/usr/bin/env python3
"""
Segment Anything Mask Processing Pipeline

This script provides an integrated pipeline that imports and calls functions directly
from the individual processing scripts, enabling efficient in-memory data passing
and optimal resource usage.

Usage:
    python scripts/pipeline.py --input <input_image_or_dir> --output <output_dir> [options]
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import functions from individual scripts
try:
    from amg import sam_model_registry, SamAutomaticMaskGenerator
    import cv2
    import torch
except ImportError as e:
    print(f"Error importing SAM dependencies: {e}")
    print("Please ensure segment-anything and dependencies are installed")
    sys.exit(1)

# Import processing functions
try:
    from filter import (
        process_mask, apply_overlap_filter, write_metadata_csv, write_contours_csv,
        find_largest_contour, calculate_circularity, is_circular,
        apply_singularity_filter, get_bounding_box
    )
    from categorise import categorize_bboxes
    from category_counting import analyze_category_containment
except ImportError as e:
    print(f"Error importing processing functions: {e}")
    print("Please ensure all script modules are available")
    sys.exit(1)


class IntegratedPipeline:
    """Integrated pipeline for SAM mask processing."""
    
    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str = "cuda"):
        """Initialize the pipeline with SAM model."""
        self.device = device
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.sam = None
        self.mask_generator = None
        
    def load_model(self, **amg_kwargs):
        """Load the SAM model and create mask generator."""
        print(f"Loading SAM model ({self.model_type}) from {self.checkpoint}...")
        try:
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            self.sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, **amg_kwargs)
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {e}")
    
    def step1_generate_masks(self, image_path: str) -> List[Dict[str, Any]]:
        """Step 1: Generate masks using SAM."""
        print(f"\n=== Step 1: Generating masks for {os.path.basename(image_path)} ===")
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        start_time = time.time()
        masks = self.mask_generator.generate(image_rgb)
        elapsed = time.time() - start_time
        
        print(f"Generated {len(masks)} masks in {elapsed:.2f} seconds")
        return masks
    
    def _process_sam_mask(self, mask_dict: Dict[str, Any], mask_id: int, 
                         circularity_threshold: float, debug: bool = False) -> Tuple[Optional[np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """
        Process a single SAM mask using the modular filter functions.
        
        Args:
            mask_dict: SAM mask dictionary
            mask_id: ID to assign to this mask
            circularity_threshold: Circularity threshold
            debug: Debug mode flag
            
        Returns:
            Tuple of (filtered_mask, metadata, contour_data)
        """
        # Convert SAM mask to binary mask
        mask = mask_dict["segmentation"].astype(np.uint8) * 255
        
        # Apply singularity filter (keep largest component)
        filtered_mask = apply_singularity_filter(mask)
        
        # Initialize metadata with SAM-specific fields
        metadata = {
            'id': mask_id,
            'filename': f"{mask_id}.png",
            'area': mask_dict.get('area', 0),
            'bbox_x0': mask_dict.get('bbox', [0, 0, 0, 0])[0],
            'bbox_y0': mask_dict.get('bbox', [0, 0, 0, 0])[1],
            'bbox_w': mask_dict.get('bbox', [0, 0, 0, 0])[2],
            'bbox_h': mask_dict.get('bbox', [0, 0, 0, 0])[3],
            'point_input_x': mask_dict.get('point_coords', [[0, 0]])[0][0],
            'point_input_y': mask_dict.get('point_coords', [[0, 0]])[0][1],
            'predicted_iou': mask_dict.get('predicted_iou', 0),
            'stability_score': mask_dict.get('stability_score', 0),
            'crop_box_x0': mask_dict.get('crop_box', [0, 0, 0, 0])[0],
            'crop_box_y0': mask_dict.get('crop_box', [0, 0, 0, 0])[1],
            'crop_box_w': mask_dict.get('crop_box', [0, 0, 0, 0])[2],
            'crop_box_h': mask_dict.get('crop_box', [0, 0, 0, 0])[3]
        }
        
        if filtered_mask is None:
            metadata.update({
                'contour_count': 0,
                'circularity': 0.0,
                'status': 'FAILED' if debug else 'FAILED',
            })
            empty_contour_data = {
                'id': mask_id,
                'filename': f"{mask_id}.png",
                'largest_contour_points': []
            }
            if debug:
                metadata['error'] = 'No valid contours found after singularity filter'
            return None, metadata, empty_contour_data
        
        # Apply circularity filter and calculate all geometric properties
        is_circ = is_circular(filtered_mask, circularity_threshold)
        
        # Calculate detailed contour information
        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity_score = 0.0
        contour_count = len(contours)
        largest_contour_points = []
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            circularity_score = calculate_circularity(largest_contour)
            # Convert contour to list of [x, y] coordinates
            largest_contour_points = largest_contour.reshape(-1, 2).tolist()
        
        # Update bounding box and area based on filtered mask (more accurate)
        bbox_x, bbox_y, bbox_w, bbox_h = get_bounding_box(filtered_mask)
        actual_area = cv2.countNonZero(filtered_mask)
        
        # Calculate centroid
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                centroid_x = M["m10"] / M["m00"]
                centroid_y = M["m01"] / M["m00"]
            else:
                centroid_x = bbox_x + bbox_w / 2
                centroid_y = bbox_y + bbox_h / 2
        else:
            centroid_x = bbox_x + bbox_w / 2
            centroid_y = bbox_y + bbox_h / 2
        
        # Update metadata with corrected geometric properties
        metadata.update({
            'area': float(actual_area),
            'bbox_x0': float(bbox_x),
            'bbox_y0': float(bbox_y), 
            'bbox_w': float(bbox_w),
            'bbox_h': float(bbox_h),
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'contour_count': contour_count,
            'circularity': round(circularity_score, 4),
            'status': 'PASSED' if is_circ else 'FAILED'
        })
        
        # Create separate contour data
        contour_data = {
            'id': mask_id,
            'filename': f"{mask_id}.png",
            'largest_contour_points': largest_contour_points
        }
        
        if debug and not is_circ:
            metadata['error'] = f'Circularity {circularity_score:.3f} below threshold {circularity_threshold}'
        
        return filtered_mask if is_circ else None, metadata, contour_data

    def step2_filter_masks(self, masks: List[Dict[str, Any]], output_dir: str,
                          overlap_threshold: float = 0.8, 
                          circularity_threshold: float = 0.6,
                          debug: bool = False) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        """Step 2: Filter masks based on circularity and overlap using modular filter functions."""
        print(f"\n=== Step 2: Filtering {len(masks)} masks ===")
        
        # Create output directory for filtered masks
        filtered_dir = os.path.join(output_dir, "filtered")
        os.makedirs(filtered_dir, exist_ok=True)
        
        # Process each mask using the modular filter functions
        mask_data_list = []
        contour_data_list = []
        
        for i, mask_dict in enumerate(masks):
            filtered_mask, metadata, contour_data = self._process_sam_mask(mask_dict, i, circularity_threshold, debug)
            
            # Collect contour data
            contour_data_list.append(contour_data)
            
            mask_data = {
                'mask': filtered_mask,
                'metadata': metadata,
                'path': os.path.join(filtered_dir, f"{i}.png")
            }
            mask_data_list.append(mask_data)
        
        # Apply overlap filter using the modular function
        print(f"Applying overlap filter (threshold: {overlap_threshold})...")
        filtered_mask_data = apply_overlap_filter(mask_data_list, overlap_threshold)
        
        # Save filtered masks and create metadata
        metadata_list = []
        saved_count = 0
        
        for mask_data in filtered_mask_data:
            metadata_list.append(mask_data['metadata'])
            
            # Save mask if it passed all filters
            if mask_data['mask'] is not None and mask_data['metadata']['status'] == 'PASSED':
                cv2.imwrite(mask_data['path'], mask_data['mask'])
                saved_count += 1
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        
        # Save metadata CSV using the modular function
        metadata_csv_path = os.path.join(filtered_dir, "metadata.csv")
        write_metadata_csv(metadata_list, Path(metadata_csv_path))
        
        # Save contours CSV
        contours_csv_path = os.path.join(filtered_dir, "contours.csv")
        write_contours_csv(contour_data_list, Path(contours_csv_path))
        
        passed_count = len(metadata_df[metadata_df['status'] == 'PASSED'])
        print(f"Filtering complete: {saved_count} masks saved, {passed_count} passed all filters")
        print(f"Contour data saved to: {contours_csv_path}")
        
        return filtered_mask_data, metadata_df
    
    def step3_categorize_masks(self, metadata_df: pd.DataFrame, 
                              overlap_threshold: float = 0.7,
                              debug: bool = False) -> pd.DataFrame:
        """Step 3: Categorize masks as parent/child."""
        print(f"\n=== Step 3: Categorizing masks ===")
        
        # Apply categorization
        categorized_df = categorize_bboxes(metadata_df, overlap_threshold, include_reason=debug)
        
        parent_count = (categorized_df['category'] == 0).sum()
        child_count = (categorized_df['category'] == 1).sum()
        
        print(f"Categorization complete:")
        print(f"  Parents (category 0): {parent_count}")
        print(f"  Children (category 1): {child_count}")
        
        return categorized_df
    
    def step4_analyze_categories(self, categorized_df: pd.DataFrame,
                                overlap_threshold: float = 0.7,
                                visualize: bool = False,
                                image_path: Optional[str] = None,
                                output_dir: Optional[str] = None,
                                filter_count: Optional[int] = None) -> Dict[str, Any]:
        """Step 4: Analyze category distribution."""
        print(f"\n=== Step 4: Analyzing category distribution ===")
        
        # Analyze containment
        distribution, results, cat1_data = analyze_category_containment(categorized_df, overlap_threshold)
        
        # Note: analyze_size_proportionality is already called within analyze_category_containment
        results_with_flags = results
        
        # Print statistics
        print(f"\nDistribution Analysis:")
        for count, freq in sorted(distribution.items()):
            print(f"  {freq} parent masks contain {count} child mask(s)")
        
        # Count disproportionate instances
        oversized = sum(1 for r in results_with_flags if r.is_oversized)
        undersized = sum(1 for r in results_with_flags if r.is_undersized)
        
        if oversized > 0:
            print(f"  {oversized} masks flagged as oversized")
        if undersized > 0:
            print(f"  {undersized} masks flagged as undersized")
        
        # Skip the old broken visualizations - we have better overlay visualizations now
        if visualize and image_path and output_dir:
            print("Note: Additional debug visualizations skipped (using improved overlay visualizations instead)")
            # The create_debug_visualizations function from category_counting.py has issues
            # because it tries to load individual mask files that don't exist in the expected location.
            # Our new overlay visualizations (masks_overlay.png and detailed_analysis.png) are much better.
        
        return {
            'distribution': distribution,
            'results': results_with_flags,
            'cat1_data': cat1_data,
            'total_parents': len([r for r in results_with_flags]),
            'total_children': sum(r.contained_count for r in results_with_flags),
            'oversized_count': oversized,
            'undersized_count': undersized
        }
    
    def create_overlay_visualization(self, image_path: str, categorized_df: pd.DataFrame, 
                                   filtered_masks_dir: str, output_dir: str):
        """Create an overlay visualization showing all filtered masks on the original image."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import to_rgba
            import random
        except ImportError:
            print("Warning: matplotlib not available for visualizations")
            return
            
        print("Creating overlay visualization...")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left: Original image
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Right: Image with all masks overlaid
        ax2.imshow(image_rgb)
        
        # Filter to only PASSED masks
        passed_masks = categorized_df[categorized_df['status'] == 'PASSED'].copy()
        
        # Create color maps for categories
        parent_color = 'red'
        child_color = 'blue'
        
        # Draw all bounding boxes
        parent_count = 0
        child_count = 0
        
        for _, row in passed_masks.iterrows():
            x, y, w, h = row['bbox_x0'], row['bbox_y0'], row['bbox_w'], row['bbox_h']
            category = row['category']
            
            if category == 0:  # Parent
                color = parent_color
                linestyle = '-'
                linewidth = 2
                alpha = 0.8
                parent_count += 1
            else:  # Child
                color = child_color
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.6
                child_count += 1
            
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, 
                                   edgecolor=color, facecolor='none', 
                                   linestyle=linestyle, alpha=alpha)
            ax2.add_patch(rect)
        
        # Add title with statistics
        title = f"Filtered Masks Overlay\n"
        title += f"Parents (Category 0): {parent_count} | Children (Category 1): {child_count}"
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add legend
        parent_patch = patches.Patch(color=parent_color, label=f'Category 0 (Parents): {parent_count}')
        child_patch = patches.Patch(color=child_color, label=f'Category 1 (Children): {child_count}')
        fig.legend(handles=[parent_patch, child_patch], loc='lower center', ncol=2, fontsize=12)
        
        # Save the visualization
        overlay_path = os.path.join(output_dir, "masks_overlay.png")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Overlay visualization saved to: {overlay_path}")
        
    def create_detailed_overlay_visualization(self, image_path: str, categorized_df: pd.DataFrame, 
                                            results_with_flags: List, cat1_data: Dict, output_dir: str):
        """Create detailed overlay visualization with containment analysis."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
        except ImportError:
            print("Warning: matplotlib not available for visualizations")
            return
            
        print("Creating detailed containment visualization...")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Original image
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 2. All masks with categories
        ax2.imshow(image_rgb)
        passed_masks = categorized_df[categorized_df['status'] == 'PASSED'].copy()
        
        parent_count = 0
        child_count = 0
        
        for _, row in passed_masks.iterrows():
            x, y, w, h = row['bbox_x0'], row['bbox_y0'], row['bbox_w'], row['bbox_h']
            mask_id = row['id']
            
            if row['category'] == 0:
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', 
                                       facecolor='none', linestyle='-', alpha=0.8)
                parent_count += 1
                # Add mask ID number for parent masks
                ax2.text(x + 5, y + 15, str(mask_id), fontsize=8, fontweight='bold', 
                        color='red', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            else:
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', 
                                       facecolor='none', linestyle='--', alpha=0.6)
                child_count += 1
                # Add mask ID number for child masks
                ax2.text(x + 5, y + 15, str(mask_id), fontsize=8, fontweight='bold', 
                        color='blue', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            ax2.add_patch(rect)
        
        ax2.set_title(f"Category Classification\nParents: {parent_count} | Children: {child_count}", 
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 3. Containment relationships
        ax3.imshow(image_rgb)
        
        # Color code by containment count
        containment_colors = plt.cm.viridis(np.linspace(0, 1, 11))  # 0-10+ colors
        
        for result in results_with_flags:
            if result.contained_count > 0:
                x, y, w, h = result.bbox
                color_idx = min(result.contained_count, 10)
                color = containment_colors[color_idx]
                
                # Draw parent with color based on containment count
                rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, 
                                       facecolor=color, alpha=0.3)
                ax3.add_patch(rect)
                
                # Add count text
                ax3.text(x + w/2, y + h/2, str(result.contained_count), 
                        ha='center', va='center', fontsize=10, fontweight='bold', 
                        color='white' if result.contained_count > 5 else 'black')
                
                # Add parent mask ID number
                ax3.text(x + 5, y + 15, str(result.cat0_id), fontsize=8, fontweight='bold', 
                        color='black', bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.9))
                
                # Draw contained children
                for cat1_id in result.contained_cat1_ids:
                    if cat1_id in cat1_data:
                        x1, y1, w1, h1 = cat1_data[cat1_id]['bbox']
                        rect = patches.Rectangle((x1, y1), w1, h1, linewidth=1, 
                                               edgecolor='yellow', facecolor='none', 
                                               linestyle=':', alpha=0.8)
                        ax3.add_patch(rect)
                        
                        # Add child mask ID number
                        ax3.text(x1 + 5, y1 + 15, str(cat1_id), fontsize=8, fontweight='bold', 
                                color='black', bbox=dict(boxstyle="round,pad=0.2", facecolor='cyan', alpha=0.9))
        
        ax3.set_title("Containment Analysis\n(Numbers show contained count)", 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # 4. Flagged instances
        ax4.imshow(image_rgb)
        
        oversized_count = 0
        undersized_count = 0
        
        for result in results_with_flags:
            x, y, w, h = result.bbox
            
            if result.is_oversized:
                rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='orange', 
                                       facecolor='orange', alpha=0.4)
                ax4.add_patch(rect)
                ax4.text(x + w/2, y + h/2, 'BIG', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='black')
                # Add mask ID number for oversized masks
                ax4.text(x + 5, y + 15, str(result.cat0_id), fontsize=8, fontweight='bold', 
                        color='black', bbox=dict(boxstyle="round,pad=0.2", facecolor='orange', alpha=0.9))
                oversized_count += 1
                
            elif result.is_undersized:
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='purple', 
                                       facecolor='purple', alpha=0.4)
                ax4.add_patch(rect)
                ax4.text(x + w/2, y + h/2, 'SMALL', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
                # Add mask ID number for undersized masks
                ax4.text(x + 5, y + 15, str(result.cat0_id), fontsize=8, fontweight='bold', 
                        color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor='purple', alpha=0.9))
                undersized_count += 1
        
        ax4.set_title(f"Size Analysis\nOversized: {oversized_count} | Undersized: {undersized_count}", 
                     fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Overall title
        fig.suptitle(f"Comprehensive Mask Analysis - {os.path.basename(image_path)}", 
                    fontsize=16, fontweight='bold')
        
        # Save the visualization
        detailed_path = os.path.join(output_dir, "detailed_analysis.png")
        plt.tight_layout()
        plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed visualization saved to: {detailed_path}")
    
    def create_unique_output_dir(self, base_output_path: str, input_path: str) -> str:
        """Create a unique output directory based on timestamp and input filename."""
        # Get input filename (without extension)
        if os.path.isdir(input_path):
            input_name = os.path.basename(input_path.rstrip('/'))
        else:
            input_name = Path(input_path).stem
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique folder name
        unique_folder = f"{input_name}_{timestamp}"
        unique_output_path = os.path.join(base_output_path, unique_folder)
        
        # Create the directory
        os.makedirs(unique_output_path, exist_ok=True)
        
        return unique_output_path

    def run_pipeline(self, input_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """Run the complete integrated pipeline."""
        start_time = time.time()
        
        # Validate inputs
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        # Create unique output directory
        unique_output_path = self.create_unique_output_dir(output_path, input_path)
        print(f"Created unique output directory: {unique_output_path}")
        
        # Handle directory vs single image
        if os.path.isdir(input_path):
            # For now, process first image in directory
            image_files = [f for f in os.listdir(input_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            if not image_files:
                raise ValueError(f"No image files found in directory: {input_path}")
            image_path = os.path.join(input_path, image_files[0])
            print(f"Processing first image from directory: {os.path.basename(image_path)}")
        else:
            image_path = input_path
        
        # Load model if not already loaded
        if self.mask_generator is None:
            amg_kwargs = {k: v for k, v in kwargs.items() 
                         if k in ['points_per_side', 'points_per_batch', 'pred_iou_thresh',
                                 'stability_score_thresh', 'stability_score_offset', 
                                 'box_nms_thresh', 'crop_n_layers', 'crop_nms_thresh',
                                 'crop_overlap_ratio', 'crop_n_points_downscale_factor',
                                 'min_mask_region_area'] and v is not None}
            self.load_model(**amg_kwargs)
        
        try:
            # Step 1: Generate masks
            masks = self.step1_generate_masks(image_path)
            
            # Step 2: Filter masks
            filtered_masks, metadata_df = self.step2_filter_masks(
                masks, unique_output_path,
                kwargs.get('overlap_threshold', 0.7),
                kwargs.get('circularity_threshold', 0.6),
                kwargs.get('debug', False)
            )
            
            # Step 3: Categorize masks
            categorized_df = self.step3_categorize_masks(
                metadata_df,
                kwargs.get('overlap_threshold', 0.7),
                kwargs.get('debug', False)
            )
            
            # Save categorized metadata
            categorized_csv = os.path.join(unique_output_path, "metadata_categorized.csv")
            categorized_df.to_csv(categorized_csv, index=False)
            print(f"Categorized metadata saved to: {categorized_csv}")
            
            # Step 4: Analyze categories
            analysis_results = self.step4_analyze_categories(
                categorized_df,
                kwargs.get('overlap_threshold', 0.7),
                kwargs.get('visualize', False),
                image_path if kwargs.get('visualize', False) else None,
                unique_output_path if kwargs.get('visualize', False) else None,
                kwargs.get('filter_count', None)
            )
            
            elapsed_time = time.time() - start_time
            
            # Create summary
            summary = {
                'pipeline_config': kwargs,
                'input_path': input_path,
                'output_path': unique_output_path,
                'base_output_path': output_path,
                'processing_time_seconds': round(elapsed_time, 2),
                'results': {
                    'total_masks_generated': len(masks),
                    'masks_after_filtering': len(metadata_df[metadata_df['status'] == 'PASSED']),
                    'parent_masks': analysis_results['total_parents'],
                    'child_masks': analysis_results['total_children'],
                    'distribution': analysis_results['distribution']
                }
            }
            
            # Save summary
            summary_file = os.path.join(unique_output_path, "pipeline_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n=== Pipeline completed successfully in {elapsed_time:.2f} seconds! ===")
            print(f"Results saved to: {unique_output_path}")
            
            # Always create overlay visualizations (they're much more useful than the original ones)
            filtered_masks_dir = os.path.join(unique_output_path, "filtered")
            self.create_overlay_visualization(image_path, categorized_df, filtered_masks_dir, unique_output_path)
            self.create_detailed_overlay_visualization(image_path, categorized_df, analysis_results['results'], analysis_results['cat1_data'], unique_output_path)
            
            return summary
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Segment Anything Mask Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (includes visualizations automatically)
  python scripts/pipeline.py --input image.jpg --output results

  # With debug mode for detailed logging
  python scripts/pipeline.py --input image.jpg --output results --debug

  # Custom checkpoint and thresholds
  python scripts/pipeline.py --input image.jpg --output results --checkpoint models/sam_vit_l_0b3195.pth --overlap-threshold 0.8 --circularity-threshold 0.7
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory of images")
    parser.add_argument("--output", type=str, required=True,
                       help="Base output directory (unique timestamped folder will be created inside)")
    parser.add_argument("--checkpoint", type=str, default="models/sam_vit_h_4b8939.pth",
                       help="Path to SAM checkpoint file (default: models/sam_vit_h_4b8939.pth)")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="vit_h",
                       choices=["default", "vit_h", "vit_l", "vit_b"],
                       help="SAM model type (default: vit_h)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (default: cuda)")
    
    # AMG arguments
    parser.add_argument("--crop-n-layers", type=int, default=3,
                       help="Number of crop layers for AMG (default: 3)")
    parser.add_argument("--points-per-side", type=int,
                       help="Points per side for mask generation")
    parser.add_argument("--pred-iou-thresh", type=float,
                       help="Predicted IoU threshold")
    parser.add_argument("--stability-score-thresh", type=float,
                       help="Stability score threshold")
    
    # Filtering arguments
    parser.add_argument("--overlap-threshold", type=float, default=0.7,
                       help="Overlap threshold for filtering and categorization (default: 0.7)")
    parser.add_argument("--circularity-threshold", type=float, default=0.6,
                       help="Circularity threshold for filtering (default: 0.6)")
    
    # Visualization arguments (kept for compatibility, but visualizations are always generated now)
    parser.add_argument("--visualize", action="store_true",
                       help="Legacy option (visualizations are now always generated)")
    parser.add_argument("--filter-count", type=int,
                       help="Legacy option (no longer used)")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode for detailed logging")
    
    args = parser.parse_args()
    
    try:
        # Create pipeline
        pipeline = IntegratedPipeline(
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            device=args.device
        )
        
        # Convert args to kwargs
        kwargs = vars(args)
        
        # Run pipeline
        summary = pipeline.run_pipeline(args.input, args.output, **kwargs)
        
        print(f"\nSummary: {summary['results']}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 