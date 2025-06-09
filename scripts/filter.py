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
import csv
from pathlib import Path
import glob
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import pandas as pd


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


def load_original_metadata(input_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load original metadata.csv if it exists in the input directory.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        DataFrame with original metadata or None if not found
    """
    metadata_path = input_dir / "metadata.csv"
    if metadata_path.exists():
        try:
            return pd.read_csv(metadata_path)
        except Exception as e:
            print(f"Warning: Could not read original metadata.csv: {e}")
            return None
    return None


def get_mask_id_from_filename(filename: str) -> Optional[int]:
    """
    Extract mask ID from filename. Assumes format like 'mask_123.png' or '123.png'
    
    Args:
        filename: Mask filename
        
    Returns:
        Mask ID as integer or None if not found
    """
    # Remove extension
    basename = Path(filename).stem
    
    # Try different patterns
    patterns = [
        basename,  # Just the number itself
        basename.replace('mask_', ''),  # Remove 'mask_' prefix
        basename.split('_')[-1],  # Last part after underscore
    ]
    
    for pattern in patterns:
        try:
            return int(pattern)
        except ValueError:
            continue
    
    return None


def process_mask(mask_path: str, circularity_threshold: float = 0.55, original_metadata: Optional[pd.DataFrame] = None, debug_mode: bool = False) -> Tuple[Optional[np.ndarray], bool, bool, float, Dict[str, Any]]:
    """
    Process a single mask through both filters.
    
    Args:
        mask_path: Path to the mask image
        circularity_threshold: Minimum circularity score
        
    Returns:
        Tuple of (filtered_mask, passed_singularity, passed_circularity, circularity_score, metadata)
    """
    filename = Path(mask_path).name
    metadata = {
        'filename': filename
    }
    
    # Try to inherit original metadata if available
    if original_metadata is not None:
        mask_id = get_mask_id_from_filename(filename)
        if mask_id is not None:
            # Find matching row in original metadata
            matching_rows = original_metadata[original_metadata['id'] == mask_id]
            if not matching_rows.empty:
                # Inherit all columns from original metadata
                original_row = matching_rows.iloc[0].to_dict()
                # Update metadata with original values, but keep our new processing-specific fields
                for key, value in original_row.items():
                    if key not in ['filename']:
                        metadata[key] = value
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load mask {mask_path}")
        if debug_mode:
            metadata.update({
                'status': 'FAILED_LOAD',
                'error': 'Could not load image'
            })
        return None, False, False, 0.0, metadata
    
    # Apply singularity filter
    singular_mask = apply_singularity_filter(mask)
    if singular_mask is None:
        if debug_mode:
            metadata.update({
                'status': 'FAILED_SINGULARITY',
                'error': 'No valid contours found'
            })
        return None, False, False, 0.0, metadata
    
    passed_singularity = True
    
    # Calculate circularity score
    contours, _ = cv2.findContours(singular_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity_score = 0.0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        circularity_score = calculate_circularity(largest_contour)
    
    # Apply circular filter
    passed_circularity = is_circular(singular_mask, circularity_threshold)
    
    if passed_circularity:
        if debug_mode:
            metadata.update({
                'status': 'PASSED',
                'error': 'none'
            })
        return singular_mask, passed_singularity, passed_circularity, circularity_score, metadata
    else:
        if debug_mode:
            metadata.update({
                'status': 'FAILED_CIRCULARITY',
                'error': f'Circularity {circularity_score:.3f} below threshold {circularity_threshold}'
            })
        return None, passed_singularity, passed_circularity, circularity_score, metadata


def write_metadata_csv(metadata_list: list, output_path: Path):
    """
    Write metadata to CSV file.
    
    Args:
        metadata_list: List of metadata dictionaries
        output_path: Path where to save the CSV file
    """
    if not metadata_list:
        return
    
    # Dynamically determine all possible columns from all metadata entries
    all_fieldnames = set()
    for metadata in metadata_list:
        all_fieldnames.update(metadata.keys())
    
    # Define preferred column order for common fields
    preferred_order = [
        'filename',
        'circularity_score',
        'circularity_threshold'
    ]
    
    # Add debug-specific columns if any metadata contains them
    if any('status' in metadata for metadata in metadata_list):
        preferred_order.insert(1, 'status')
    
    # Create final fieldnames list: preferred order first, then any additional fields
    fieldnames = []
    for field in preferred_order:
        if field in all_fieldnames:
            fieldnames.append(field)
            all_fieldnames.remove(field)
    
    # Add any remaining fields that weren't in the preferred order (except error)
    remaining_fields = [field for field in sorted(all_fieldnames) if field != 'error']
    fieldnames.extend(remaining_fields)
    
    # Add error column at the end for visibility
    if 'error' in all_fieldnames:
        fieldnames.append('error')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for metadata in metadata_list:
            # Write all available metadata for this entry
            writer.writerow(metadata)


def main():
    parser = argparse.ArgumentParser(description="Filter masks using singularity and circular filters. If metadata.csv exists in input directory, all original columns will be inherited in the output metadata.")
    parser.add_argument("input_dir", help="Directory containing input masks")
    parser.add_argument("-o", "--output_dir", help="Output directory for filtered masks (default: creates filter_TIMESTAMP folder inside input_dir)")
    parser.add_argument("-c", "--circularity_threshold", type=float, default=0.55, 
                       help="Minimum circularity score (0.0-1.0, default: 0.55)")
    parser.add_argument("-e", "--extensions", nargs="+", default=["png", "jpg", "jpeg", "bmp", "tiff"],
                       help="Image file extensions to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input directory instead of creating new folder")
    parser.add_argument("--dry_run", action="store_true", help="Show statistics without saving filtered masks")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: show detailed logs and include all masks in metadata (passed and failed)")
    
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
    
    # Load original metadata if available
    original_metadata = load_original_metadata(input_dir)
    if original_metadata is not None:
        print(f"Found original metadata.csv with {len(original_metadata)} entries")
    else:
        print("No original metadata.csv found - creating new metadata from scratch")
    
    # Process masks
    stats = {
        'total': len(mask_files),
        'passed_singularity': 0,
        'passed_circularity': 0,
        'passed_both': 0,
        'failed_load': 0
    }
    
    metadata_list = []
    
    for i, mask_path in enumerate(mask_files):
        if args.debug:
            print(f"Processing {i+1}/{len(mask_files)}: {Path(mask_path).name}", end=" ")
        
        filtered_mask, passed_sing, passed_circ, circularity_score, metadata = process_mask(mask_path, args.circularity_threshold, original_metadata, args.debug)
        
        # Add additional metadata
        metadata.update({
            'circularity_score': round(circularity_score, 4),
            'circularity_threshold': args.circularity_threshold
        })
        
        # In debug mode, include all masks; otherwise only include passed masks
        if args.debug or (filtered_mask is not None and passed_circ):
            metadata_list.append(metadata)
        
        if filtered_mask is None and not passed_sing:
            stats['failed_load'] += 1
            if args.debug:
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
            if args.debug:
                print("- PASSED")
        else:
            if args.debug:
                if passed_sing and not passed_circ:
                    print("- FAILED (not circular enough)")
                else:
                    print("- FAILED")
    
    # Write metadata CSV
    if not args.dry_run:
        metadata_csv_path = output_dir / "metadata.csv"
        write_metadata_csv(metadata_list, metadata_csv_path)
        print(f"\nMetadata saved to: {metadata_csv_path}")
    
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
