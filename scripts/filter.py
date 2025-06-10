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
from typing import Tuple, Optional, Dict, Any, List
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


def get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box coordinates from a binary mask.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Tuple of (x, y, width, height) of bounding box
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, 0, 0)
    
    # Get bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_contour)


def calculate_bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1: First bounding box as (x, y, width, height)
        bbox2: Second bounding box as (x, y, width, height)
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def apply_overlap_filter(mask_data_list: List[Dict[str, Any]], overlap_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Apply overlap filter to remove masks with high bounding box overlap.
    
    Args:
        mask_data_list: List of dictionaries containing mask data and metadata
        overlap_threshold: IoU threshold above which masks are considered overlapping
        
    Returns:
        Filtered list with overlapping masks removed (keeps the one with highest circularity)
    """
    if len(mask_data_list) <= 1:
        return mask_data_list
    
    # Calculate bounding boxes for all masks
    for mask_data in mask_data_list:
        if mask_data['mask'] is not None:
            mask_data['bbox'] = get_bounding_box(mask_data['mask'])
        else:
            mask_data['bbox'] = (0, 0, 0, 0)
    
    # Find groups of overlapping masks
    to_remove = set()
    
    for i in range(len(mask_data_list)):
        if i in to_remove:
            continue
            
        for j in range(i + 1, len(mask_data_list)):
            if j in to_remove:
                continue
            
            # Skip if either mask is None
            if mask_data_list[i]['mask'] is None or mask_data_list[j]['mask'] is None:
                continue
            
            # Calculate IoU
            iou = calculate_bbox_iou(mask_data_list[i]['bbox'], mask_data_list[j]['bbox'])
            
            if iou >= overlap_threshold:
                # Remove the larger mask (keep the smaller one)
                area_i = mask_data_list[i]['metadata'].get('area', 0.0)
                area_j = mask_data_list[j]['metadata'].get('area', 0.0)
                
                filename_i = Path(mask_data_list[i]['path']).name
                filename_j = Path(mask_data_list[j]['path']).name
                
                if area_i >= area_j:
                    # Remove the larger mask (i)
                    to_remove.add(i)
                    mask_data_list[i]['metadata']['overlap_iou'] = round(iou, 4)
                    mask_data_list[i]['metadata']['overlap_with'] = filename_j
                    break  # Move to next i since this one is being removed
                else:
                    # Remove the larger mask (j)
                    to_remove.add(j)
                    mask_data_list[j]['metadata']['overlap_iou'] = round(iou, 4)
                    mask_data_list[j]['metadata']['overlap_with'] = filename_i
    
    # Return filtered list
    filtered_list = []
    for i, mask_data in enumerate(mask_data_list):
        if i not in to_remove:
            filtered_list.append(mask_data)
    
    return filtered_list


def is_circular(mask: np.ndarray, circularity_threshold: float = 0.6) -> bool:
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


def touches_edge(bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int], edge_margin: int = 0) -> bool:
    """
    Check if a bounding box touches the edge of an image.
    
    Args:
        bbox: Bounding box as (x, y, width, height)
        image_shape: Image shape as (height, width)
        edge_margin: Minimum distance from edge (default: 0 means touching edge fails)
        
    Returns:
        True if bounding box touches edge (within margin), False otherwise
    """
    x, y, w, h = bbox
    img_height, img_width = image_shape
    
    # Check if any edge of the bounding box is too close to image edge
    left_edge = x <= edge_margin
    top_edge = y <= edge_margin
    right_edge = (x + w) >= (img_width - edge_margin)
    bottom_edge = (y + h) >= (img_height - edge_margin)
    
    return left_edge or top_edge or right_edge or bottom_edge


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


def process_mask(mask_path: str, circularity_threshold: float = 0.6, original_metadata: Optional[pd.DataFrame] = None, debug_mode: bool = False, enable_edge_filter: bool = True, edge_margin: int = 0) -> Tuple[Optional[np.ndarray], bool, bool, bool, float, Dict[str, Any], Dict[str, Any]]:
    """
    Process a single mask through all filters.
    
    Args:
        mask_path: Path to the mask image
        circularity_threshold: Minimum circularity score
        original_metadata: Optional DataFrame with original metadata to inherit from
        debug_mode: Enable debug mode for detailed error information
        enable_edge_filter: Enable edge detection filter
        edge_margin: Minimum distance from edge (pixels)
        
    Returns:
        Tuple of (filtered_mask, passed_singularity, passed_circularity, passed_edge, circularity_score, metadata, contour_data)
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
        empty_contour_data = {
            'id': metadata.get('id', get_mask_id_from_filename(filename)),
            'filename': filename,
            'largest_contour_points': []
        }
        if debug_mode:
            metadata.update({
                'status': 'FAILED_LOAD',
                'error': 'Could not load image'
            })
        return None, False, False, False, 0.0, metadata, empty_contour_data
    
    # Apply singularity filter
    singular_mask = apply_singularity_filter(mask)
    if singular_mask is None:
        empty_contour_data = {
            'id': metadata.get('id', get_mask_id_from_filename(filename)),
            'filename': filename,
            'largest_contour_points': []
        }
        if debug_mode:
            metadata.update({
                'status': 'FAILED_SINGULARITY',
                'error': 'No valid contours found'
            })
        return None, False, False, False, 0.0, metadata, empty_contour_data
    
    passed_singularity = True
    
    # Calculate circularity score and extract contour points
    contours, _ = cv2.findContours(singular_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity_score = 0.0
    contour_count = len(contours)
    largest_contour_points = []
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        circularity_score = calculate_circularity(largest_contour)
        # Convert contour to list of [x, y] coordinates
        largest_contour_points = largest_contour.reshape(-1, 2).tolist()
    
    # Update bounding box coordinates based on the filtered mask
    bbox_x, bbox_y, bbox_w, bbox_h = get_bounding_box(singular_mask)
    
    # Apply edge filter
    passed_edge = True
    if enable_edge_filter:
        image_shape = mask.shape  # Original mask shape (height, width)
        bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
        if touches_edge(bbox, image_shape, edge_margin):
            passed_edge = False
    
    # Calculate centroid from the filtered mask
    contours, _ = cv2.findContours(singular_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    
    # Calculate actual area from the filtered mask
    actual_area = cv2.countNonZero(singular_mask)
    
    # Update metadata with corrected geometric properties
    metadata.update({
        'area': float(actual_area),
        'bbox_w': float(bbox_w),
        'bbox_h': float(bbox_h), 
        'bbox_x0': float(bbox_x),
        'bbox_y0': float(bbox_y),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'contour_count': contour_count,
        'circularity': round(circularity_score, 4),
        'touches_edge': not passed_edge,
        'edge_margin': edge_margin
    })
    
    # Create separate contour data for dedicated contours file
    contour_data = {
        'id': metadata.get('id', get_mask_id_from_filename(filename)),
        'filename': filename,
        'largest_contour_points': largest_contour_points
    }
    
    # Apply circular filter
    passed_circularity = is_circular(singular_mask, circularity_threshold)
    
    # Check if mask passes all filters
    passed_all_filters = passed_circularity and passed_edge
    
    if passed_all_filters:
        if debug_mode:
            metadata.update({
                'status': 'PASSED',
                'error': 'none'
            })
        return singular_mask, passed_singularity, passed_circularity, passed_edge, circularity_score, metadata, contour_data
    else:
        if debug_mode:
            if not passed_circularity and not passed_edge:
                metadata.update({
                    'status': 'FAILED_CIRCULARITY_AND_EDGE',
                    'error': f'Circularity {circularity_score:.3f} below threshold {circularity_threshold} and touches edge'
                })
            elif not passed_circularity:
                metadata.update({
                    'status': 'FAILED_CIRCULARITY',
                    'error': f'Circularity {circularity_score:.3f} below threshold {circularity_threshold}'
                })
            else:  # not passed_edge
                metadata.update({
                    'status': 'FAILED_EDGE',
                    'error': f'Bounding box touches image edge (margin: {edge_margin}px)'
                })
        return None, passed_singularity, passed_circularity, passed_edge, circularity_score, metadata, contour_data


def write_contours_csv(contours_list: list, output_path: Path):
    """
    Write contour data to CSV file.
    
    Args:
        contours_list: List of contour data dictionaries
        output_path: Path where to save the CSV file
    """
    if not contours_list:
        return
    
    fieldnames = ['id', 'filename', 'largest_contour_points']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for contour_data in contours_list:
            writer.writerow(contour_data)


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
        'id',
        'area',
        'bbox_x0',
        'bbox_y0', 
        'bbox_w',
        'bbox_h',
        'centroid_x',
        'centroid_y',
        'contour_count',
        'circularity',
        'circularity_score',
        'circularity_threshold',
        'touches_edge',
        'edge_margin',
        'edge_filter_enabled',
        'overlap_threshold',
        'overlap_iou',
        'overlap_with'
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
    parser = argparse.ArgumentParser(description="Filter masks using singularity, circular, edge, and overlap filters. If metadata.csv exists in input directory, all original columns will be inherited in the output metadata.")
    parser.add_argument("input_dir", help="Directory containing input masks")
    parser.add_argument("-o", "--output_dir", help="Output directory for filtered masks (default: creates filter_TIMESTAMP folder inside input_dir)")
    parser.add_argument("-c", "--circularity_threshold", type=float, default=0.6, 
                       help="Minimum circularity score (0.0-1.0, default: 0.6)")
    parser.add_argument("--overlap_threshold", type=float, default=0.8,
                       help="IoU threshold for overlap filter (0.0-1.0, default: 0.8)")
    parser.add_argument("--disable_edge_filter", action="store_true", default=False,
                       help="Disable edge detection filter (allow masks touching image edges)")
    parser.add_argument("--edge_margin", type=int, default=0,
                       help="Minimum distance from image edge in pixels (default: 0)")
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
    print(f"Overlap threshold: {args.overlap_threshold}")
    print(f"Edge filter: {'disabled' if args.disable_edge_filter else 'enabled'}")
    if not args.disable_edge_filter:
        print(f"Edge margin: {args.edge_margin} pixels")
    
    # Load original metadata if available
    original_metadata = load_original_metadata(input_dir)
    if original_metadata is not None:
        print(f"Found original metadata.csv with {len(original_metadata)} entries")
    else:
        print("No original metadata.csv found - creating new metadata from scratch")
    
    # Process masks (singularity + circularity + edge filters)
    stats = {
        'total': len(mask_files),
        'passed_singularity': 0,
        'passed_circularity': 0,
        'passed_edge': 0,
        'passed_all_initial': 0,
        'passed_overlap': 0,
        'failed_load': 0
    }
    
    all_metadata_list = []  # For debug mode - includes all masks
    all_contours_list = []  # For contour data collection
    passed_mask_data_list = []  # For overlap filtering - only passed masks
    
    for i, mask_path in enumerate(mask_files):
        if args.debug:
            print(f"Processing {i+1}/{len(mask_files)}: {Path(mask_path).name}", end=" ")
        
        filtered_mask, passed_sing, passed_circ, passed_edge, circularity_score, metadata, contour_data = process_mask(
            mask_path, 
            args.circularity_threshold, 
            original_metadata, 
            args.debug,
            not args.disable_edge_filter,
            args.edge_margin
        )
        
        # Always collect contour data
        all_contours_list.append(contour_data)
        
        # Add additional metadata
        metadata.update({
            'circularity_score': round(circularity_score, 4),
            'circularity_threshold': args.circularity_threshold,
            'overlap_threshold': args.overlap_threshold,
            'edge_filter_enabled': not args.disable_edge_filter
        })
        
        # Always add to all_metadata_list for debug mode
        if args.debug:
            all_metadata_list.append(metadata)
        
        if filtered_mask is None and not passed_sing:
            stats['failed_load'] += 1
            if args.debug:
                print("- FAILED (could not load/process)")
            continue
        
        if passed_sing:
            stats['passed_singularity'] += 1
        
        if passed_circ:
            stats['passed_circularity'] += 1
            
        if passed_edge:
            stats['passed_edge'] += 1
        
        if filtered_mask is not None and passed_circ and passed_edge:
            stats['passed_all_initial'] += 1
            
            # Store mask data for overlap filtering
            passed_mask_data_list.append({
                'mask': filtered_mask,
                'metadata': metadata,
                'path': mask_path
            })
            
            if args.debug:
                print("- PASSED (before overlap filter)")
        else:
            if args.debug:
                failed_reasons = []
                if not passed_sing:
                    failed_reasons.append("singularity")
                if not passed_circ:
                    failed_reasons.append("circularity")
                if not passed_edge:
                    failed_reasons.append("edge")
                
                if failed_reasons:
                    print(f"- FAILED ({', '.join(failed_reasons)})")
                else:
                    print("- FAILED")
    
    # Apply overlap filter to passed masks
    print(f"\nApplying overlap filter to {len(passed_mask_data_list)} masks...")
    filtered_mask_data_list = apply_overlap_filter(passed_mask_data_list, args.overlap_threshold)
    stats['passed_overlap'] = len(filtered_mask_data_list)
    
    # Create final metadata list
    if args.debug:
        # In debug mode, use all_metadata_list but update overlap info for removed masks
        final_metadata_list = all_metadata_list
        
        # Create mapping from filename to updated metadata from overlap filter
        overlap_metadata_map = {}
        for data in passed_mask_data_list:
            filename = Path(data['path']).name
            overlap_metadata_map[filename] = data['metadata']
        
        # Add overlap info to passed masks
        kept_filenames = {Path(data['path']).name for data in filtered_mask_data_list}
        for metadata in final_metadata_list:
            filename = metadata['filename']
            
            if filename in kept_filenames:
                # This mask was kept after overlap filter - no changes needed
                pass
            elif metadata.get('status') in ['PASSED']:
                # This mask passed initial filters but was removed by overlap filter
                metadata['status'] = 'FAILED_OVERLAP'
                if filename in overlap_metadata_map:
                    overlap_meta = overlap_metadata_map[filename]
                    if 'overlap_iou' in overlap_meta:
                        metadata['overlap_iou'] = overlap_meta['overlap_iou']
                    if 'overlap_with' in overlap_meta:
                        overlap_with = overlap_meta['overlap_with']
                        iou_value = overlap_meta.get('overlap_iou', 'unknown')
                        metadata['error'] = f'Overlaps with {overlap_with} (IoU: {iou_value})'
                    else:
                        metadata['error'] = 'Removed due to overlap with another mask'
                else:
                    metadata['error'] = 'Removed due to overlap with another mask'
    else:
        # In normal mode, only include masks that passed all filters
        final_metadata_list = [data['metadata'] for data in filtered_mask_data_list]
    
    # Save filtered masks
    if not args.dry_run:
        for mask_data in filtered_mask_data_list:
            output_path = output_dir / Path(mask_data['path']).name
            cv2.imwrite(str(output_path), mask_data['mask'])
    
    # Write metadata CSV
    if not args.dry_run:
        metadata_csv_path = output_dir / "metadata.csv"
        write_metadata_csv(final_metadata_list, metadata_csv_path)
        print(f"Metadata saved to: {metadata_csv_path}")
        
        # Write contours CSV
        contours_csv_path = output_dir / "contours.csv"
        write_contours_csv(all_contours_list, contours_csv_path)
        print(f"Contour data saved to: {contours_csv_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("FILTERING STATISTICS")
    print("="*50)
    print(f"Total masks processed: {stats['total']}")
    print(f"Failed to load/process: {stats['failed_load']}")
    print(f"Passed singularity filter: {stats['passed_singularity']}")
    print(f"Passed circularity filter: {stats['passed_circularity']}")
    print(f"Passed edge filter: {stats['passed_edge']}")
    print(f"Passed all initial filters: {stats['passed_all_initial']}")
    print(f"Passed all filters (including overlap): {stats['passed_overlap']}")
    print(f"Removed by overlap filter: {stats['passed_all_initial'] - stats['passed_overlap']}")
    print(f"Final success rate: {stats['passed_overlap']/stats['total']*100:.1f}%")
    
    if not args.dry_run:
        print(f"\nFiltered masks saved to: {output_dir}")


if __name__ == "__main__":
    main()
