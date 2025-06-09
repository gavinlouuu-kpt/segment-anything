#!/usr/bin/env python3
"""
Categorize parent-child relationships based on bounding box containment.

This script analyzes metadata CSV files containing bounding box coordinates (xywh format)
and categorizes each bbox as:
- Parent (0): Not contained by any other bbox
- Child (1): Contained by another bbox with >70% overlap

Usage:
    python categorise.py <metadata_csv_path> [output_csv_path] [overlap_threshold] [--debug]
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Tuple, List


def calculate_overlap_percentage(bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the percentage of bbox1 that overlaps with bbox2.
    
    Args:
        bbox1: (x, y, w, h) of the first bounding box
        bbox2: (x, y, w, h) of the second bounding box
        
    Returns:
        Percentage of bbox1 that overlaps with bbox2 (0.0 to 1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate the coordinates of the intersection rectangle
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    
    # If there's no intersection
    if left >= right or top >= bottom:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (right - left) * (bottom - top)
    
    # Calculate bbox1 area
    bbox1_area = w1 * h1
    
    # Return overlap percentage
    return intersection_area / bbox1_area if bbox1_area > 0 else 0.0


def is_contained_by(small_bbox: Tuple[int, int, int, int], 
                   large_bbox: Tuple[int, int, int, int], 
                   overlap_threshold: float = 0.7) -> bool:
    """
    Check if small_bbox is contained by large_bbox with sufficient overlap.
    
    Args:
        small_bbox: (x, y, w, h) of the potentially contained bbox
        large_bbox: (x, y, w, h) of the potentially containing bbox
        overlap_threshold: Minimum overlap percentage required (default 0.7 = 70%)
        
    Returns:
        True if small_bbox is contained by large_bbox with >overlap_threshold overlap
    """
    # Check if large_bbox is actually larger in area
    small_area = small_bbox[2] * small_bbox[3]
    large_area = large_bbox[2] * large_bbox[3]
    
    if large_area <= small_area:
        return False
    
    # Calculate overlap percentage
    overlap_pct = calculate_overlap_percentage(small_bbox, large_bbox)
    
    return overlap_pct > overlap_threshold


def categorize_bboxes(df: pd.DataFrame, overlap_threshold: float = 0.7, include_reason: bool = False) -> pd.DataFrame:
    """
    Categorize bounding boxes as parent (0) or child (1) based on containment.
    Only considers masks with PASSED status for categorization logic.
    
    Args:
        df: DataFrame with columns 'bbox_x0', 'bbox_y0', 'bbox_w', 'bbox_h'
        overlap_threshold: Minimum overlap percentage for containment (default 0.7)
        include_reason: Whether to include categorization reason in metadata
        
    Returns:
        DataFrame with added 'category' column (0=parent, 1=child) and optionally 'category_reason'
    """
    df = df.copy()
    df['category'] = 0  # Initialize all as parent (0)
    
    if include_reason:
        df['category_reason'] = "Not contained by any other bbox"  # Default reason for parents
    
    # Filter to only PASSED masks for categorization logic
    if 'status' in df.columns:
        passed_mask = df['status'] == 'PASSED'
        passed_df = df[passed_mask].copy()
        print(f"Filtering to {len(passed_df)} PASSED masks out of {len(df)} total masks for categorization")
        
        # Set FAILED masks to have a specific reason if include_reason is True
        if include_reason:
            failed_mask = df['status'] != 'PASSED'
            df.loc[failed_mask, 'category_reason'] = "Excluded from categorization (FAILED status)"
    else:
        # If no status column, assume all are passed
        passed_df = df.copy()
        print(f"No status column found, assuming all {len(df)} masks are passed")
    
    n_bboxes = len(passed_df)
    
    if n_bboxes == 0:
        print("No PASSED masks found for categorization")
        return df
    
    # Extract bbox coordinates as tuples for faster processing (only from PASSED masks)
    bboxes = [(row['bbox_x0'], row['bbox_y0'], row['bbox_w'], row['bbox_h']) 
              for _, row in passed_df.iterrows()]
    
    # Get bbox IDs (assuming 'id' column exists, otherwise use index)
    if 'id' in passed_df.columns:
        bbox_ids = passed_df['id'].tolist()
    else:
        bbox_ids = list(range(len(passed_df)))
    
    # Get the original indices in the full dataframe
    passed_indices = passed_df.index.tolist()
    
    print(f"Processing {n_bboxes} PASSED bounding boxes...")
    
    # For each PASSED bbox, check if it's contained by any other PASSED bbox
    for i in range(n_bboxes):
        current_bbox = bboxes[i]
        current_id = bbox_ids[i]
        current_df_idx = passed_indices[i]
        
        for j in range(n_bboxes):
            if i == j:  # Skip self-comparison
                continue
                
            potential_parent = bboxes[j]
            parent_id = bbox_ids[j]
            parent_df_idx = passed_indices[j]
            
            # Check if current bbox is contained by potential parent
            if is_contained_by(current_bbox, potential_parent, overlap_threshold):
                df.iloc[df.index.get_loc(current_df_idx), df.columns.get_loc('category')] = 1  # Mark as child
                
                if include_reason:
                    overlap_pct = calculate_overlap_percentage(current_bbox, potential_parent)
                    reason = f"Contained by bbox {parent_id} with {overlap_pct*100:.1f}% overlap"
                    df.iloc[df.index.get_loc(current_df_idx), df.columns.get_loc('category_reason')] = reason
                
                print(f"  Bbox {current_id} (area={current_bbox[2]*current_bbox[3]}) is contained by "
                      f"Bbox {parent_id} (area={potential_parent[2]*potential_parent[3]})")
                break  # Once we find a parent, we can stop checking
    
    return df


def print_statistics(df: pd.DataFrame):
    """Print categorization statistics."""
    parent_count = (df['category'] == 0).sum()
    child_count = (df['category'] == 1).sum()
    total = len(df)
    
    print(f"\nCategorization Results:")
    print(f"  Total bounding boxes: {total}")
    print(f"  Parents (category 0): {parent_count} ({parent_count/total*100:.1f}%)")
    print(f"  Children (category 1): {child_count} ({child_count/total*100:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python categorise.py <metadata_csv_path> [output_csv_path] [overlap_threshold] [--debug]")
        print("Example: python categorise.py output/fluor_bead/metadata.csv output_categorized.csv 0.7")
        print("Example with reason: python categorise.py output/fluor_bead/metadata.csv output_categorized.csv 0.7 --debug")
        print("Note: Only PASSED masks are considered for categorization. FAILED masks are excluded.")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    # Parse arguments for overlap threshold and include reason flag
    overlap_threshold = 0.7
    include_reason = False
    
    for arg in sys.argv[2:]:
        if arg == '--debug':
            include_reason = True
        elif not arg.startswith('--') and arg != output_path:
            try:
                overlap_threshold = float(arg)
            except ValueError:
                pass
    
    # Validate input file
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)
    
    print(f"Loading metadata from: {input_path}")
    print(f"Overlap threshold: {overlap_threshold*100:.1f}%")
    print(f"Include categorization reason: {include_reason}")
    
    try:
        # Load the metadata CSV
        df = pd.read_csv(input_path)
        
        # Validate required columns
        required_cols = ['bbox_x0', 'bbox_y0', 'bbox_w', 'bbox_h']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        print(f"Loaded {len(df)} bounding boxes from metadata file.")
        
        # Categorize the bounding boxes
        df_categorized = categorize_bboxes(df, overlap_threshold, include_reason)
        
        # Print statistics
        print_statistics(df_categorized)
        
        # Save results
        if output_path:
            df_categorized.to_csv(output_path, index=False)
            print(f"\nCategorized metadata saved to: {output_path}")
        else:
            # Overwrite the original file
            df_categorized.to_csv(input_path, index=False)
            print(f"\nCategorized metadata saved to: {input_path} (original file overwritten)")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
