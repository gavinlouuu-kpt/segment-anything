#!/usr/bin/env python3
"""
Mask filtering script that applies:
1. Singularity filter: keeps only the largest blob in masks with multiple blobs
2. Circular filter: filters out masks that are not circular using OpenCV
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import glob
from typing import Tuple, Optional


def find_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest contour in a binary mask.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Mask with only the largest connected component, or None if no contours found
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a new mask with only the largest contour
    filtered_mask = np.zeros_like(mask)
    cv2.fillPoly(filtered_mask, [largest_contour], 255)
    
    return filtered_mask


def calculate_circularity(contour: np.ndarray) -> float:
    """
    Calculate the circularity of a contour.
    Circularity = 4π * area / perimeter²
    Perfect circle has circularity = 1.0
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Circularity score (0.0 to 1.0)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0.0
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return min(circularity, 1.0)  # Cap at 1.0 for numerical stability


def is_circular(mask: np.ndarray, circularity_threshold: float = 0.7) -> bool:
    """
    Check if a mask represents a circular shape.
    
    Args:
        mask: Binary mask as numpy array
        circularity_threshold: Minimum circularity score to be considered circular
        
    Returns:
        True if the mask is circular enough, False otherwise
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Get the largest contour (should be the main shape after singularity filter)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate circularity
    circularity = calculate_circularity(largest_contour)
    
    return circularity >= circularity_threshold


def apply_singularity_filter(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply singularity filter: keep only the largest blob.
    
    Args:
        mask: Input binary mask
        
    Returns:
        Filtered mask with only the largest blob, or None if no valid contours
    """
    if mask is None or mask.size == 0:
        return None
    
    # Ensure mask is binary
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Threshold to ensure binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return find_largest_contour(binary_mask)


def process_mask(mask_path: str, circularity_threshold: float = 0.7) -> Tuple[Optional[np.ndarray], bool, bool]:
    """
    Process a single mask through both filters.
    
    Args:
        mask_path: Path to the mask image
        circularity_threshold: Minimum circularity score
        
    Returns:
        Tuple of (filtered_mask, passed_singularity, passed_circularity)
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load mask {mask_path}")
        return None, False, False
    
    # Apply singularity filter
    singular_mask = apply_singularity_filter(mask)
    if singular_mask is None:
        return None, False, False
    
    passed_singularity = True
    
    # Apply circular filter
    passed_circularity = is_circular(singular_mask, circularity_threshold)
    
    if passed_circularity:
        return singular_mask, passed_singularity, passed_circularity
    else:
        return None, passed_singularity, passed_circularity


def main():
    parser = argparse.ArgumentParser(description="Filter masks using singularity and circular filters")
    parser.add_argument("input_dir", help="Directory containing input masks")
    parser.add_argument("-o", "--output_dir", help="Output directory for filtered masks (default: creates filter_TIMESTAMP folder inside input_dir)")
    parser.add_argument("-c", "--circularity_threshold", type=float, default=0.7, 
                       help="Minimum circularity score (0.0-1.0, default: 0.7)")
    parser.add_argument("-e", "--extensions", nargs="+", default=["png", "jpg", "jpeg", "bmp", "tiff"],
                       help="Image file extensions to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input directory instead of creating new folder")
    parser.add_argument("--dry_run", action="store_true", help="Show statistics without saving filtered masks")
    
    args = parser.parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    if args.overwrite:
        output_dir = input_dir
        print("Warning: Overwrite mode enabled - filtered masks will replace original masks")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create filter_TIMESTAMP folder inside input directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_dir / f"filter_{timestamp}"
    
    if not args.dry_run:
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Find all mask files
    mask_files = []
    for ext in args.extensions:
        mask_files.extend(glob.glob(str(input_dir / f"*.{ext}")))
        mask_files.extend(glob.glob(str(input_dir / f"*.{ext.upper()}")))
    
    if not mask_files:
        print(f"No mask files found in {input_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files")
    print(f"Circularity threshold: {args.circularity_threshold}")
    
    # Process masks
    stats = {
        'total': len(mask_files),
        'passed_singularity': 0,
        'passed_circularity': 0,
        'passed_both': 0,
        'failed_load': 0
    }
    
    for i, mask_path in enumerate(mask_files):
        print(f"Processing {i+1}/{len(mask_files)}: {Path(mask_path).name}", end=" ")
        
        filtered_mask, passed_sing, passed_circ = process_mask(mask_path, args.circularity_threshold)
        
        if filtered_mask is None and not passed_sing:
            stats['failed_load'] += 1
            print("- FAILED (could not load/process)")
            continue
        
        if passed_sing:
            stats['passed_singularity'] += 1
        
        if passed_circ:
            stats['passed_circularity'] += 1
        
        if filtered_mask is not None and passed_circ:
            stats['passed_both'] += 1
            
            if not args.dry_run:
                # Save filtered mask
                output_path = output_dir / Path(mask_path).name
                cv2.imwrite(str(output_path), filtered_mask)
            print("- PASSED")
        else:
            if passed_sing and not passed_circ:
                print("- FAILED (not circular enough)")
            else:
                print("- FAILED")
    
    # Print statistics
    print("\n" + "="*50)
    print("FILTERING STATISTICS")
    print("="*50)
    print(f"Total masks processed: {stats['total']}")
    print(f"Failed to load/process: {stats['failed_load']}")
    print(f"Passed singularity filter: {stats['passed_singularity']}")
    print(f"Passed circularity filter: {stats['passed_circularity']}")
    print(f"Passed both filters: {stats['passed_both']}")
    print(f"Success rate: {stats['passed_both']/stats['total']*100:.1f}%")
    
    if not args.dry_run:
        print(f"\nFiltered masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
