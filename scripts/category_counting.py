#!/usr/bin/env python3
"""
Analyze category distribution by counting how many category 1 instances 
are contained within each category 0 instance.

This script examines metadata CSV files and for each category 0 instance:
- Counts how many category 1 instances have bounding boxes contained within it
- Reports distribution: how many cat 0 contain 0 cat 1, 1 cat 1, 2 cat 1, etc.
- Flags instances with disproportionate size relative to containment count
- Provides integrated debug visualizations

Usage:
    python category_counting.py <metadata_csv_path> [overlap_threshold] [--visualize] [--image-dir <path>] [--output-dir <path>] [--filter-count <n>]
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
from typing import Tuple, List, Dict, NamedTuple, Optional
from collections import defaultdict
from pathlib import Path

# Optional imports for visualization
try:
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ContainmentResult(NamedTuple):
    """Result of containment analysis for a single category 0 instance."""
    cat0_id: int
    cat0_area: float
    contained_count: int
    contained_cat1_ids: List[int]
    area_per_contained: float
    is_oversized: bool
    is_undersized: bool
    filename: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


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


def analyze_category_containment(df: pd.DataFrame, overlap_threshold: float = 0.7) -> Tuple[Dict[int, int], List[ContainmentResult]]:
    """
    Analyze how many category 1 instances are contained within each category 0 instance.
    
    Args:
        df: DataFrame with columns 'bbox_x0', 'bbox_y0', 'bbox_w', 'bbox_h', 'category'
        overlap_threshold: Minimum overlap percentage for containment (default 0.7)
        
    Returns:
        Tuple of:
        - Dictionary mapping number of contained cat1 instances to count of cat0 instances
        - List of detailed containment results for each category 0 instance
    """
    # Filter to only PASSED masks
    if 'status' in df.columns:
        passed_mask = df['status'] == 'PASSED'
        df_filtered = df[passed_mask].copy()
        print(f"Filtering to {len(df_filtered)} PASSED masks out of {len(df)} total masks")
    else:
        df_filtered = df.copy()
        print(f"No status column found, assuming all {len(df)} masks are passed")
    
    # Separate category 0 and category 1 instances
    cat0_df = df_filtered[df_filtered['category'] == 0].copy()
    cat1_df = df_filtered[df_filtered['category'] == 1].copy()
    
    print(f"Found {len(cat0_df)} category 0 instances and {len(cat1_df)} category 1 instances")
    
    if len(cat0_df) == 0:
        print("No category 0 instances found!")
        return {}, []
    
    if len(cat1_df) == 0:
        print("No category 1 instances found!")
        empty_results = [
            ContainmentResult(
                cat0_id=row['id'] if 'id' in cat0_df.columns else i,
                cat0_area=row['bbox_w'] * row['bbox_h'],
                contained_count=0,
                contained_cat1_ids=[],
                area_per_contained=float('inf'),
                is_oversized=False,
                is_undersized=False,
                filename=row.get('filename', ''),
                bbox=(row['bbox_x0'], row['bbox_y0'], row['bbox_w'], row['bbox_h'])
            )
            for i, (_, row) in enumerate(cat0_df.iterrows())
        ]
        return {0: len(cat0_df)}, empty_results
    
    # Convert to lists of tuples for faster processing
    cat0_bboxes = [(row['bbox_x0'], row['bbox_y0'], row['bbox_w'], row['bbox_h']) 
                   for _, row in cat0_df.iterrows()]
    cat1_bboxes = [(row['bbox_x0'], row['bbox_y0'], row['bbox_w'], row['bbox_h']) 
                   for _, row in cat1_df.iterrows()]
    
    cat0_ids = cat0_df['id'].tolist() if 'id' in cat0_df.columns else list(range(len(cat0_df)))
    cat1_ids = cat1_df['id'].tolist() if 'id' in cat1_df.columns else list(range(len(cat1_df)))
    
    # Create mapping for cat1 data
    cat1_data = {}
    for idx, (_, row) in enumerate(cat1_df.iterrows()):
        cat1_id = cat1_ids[idx]
        cat1_data[cat1_id] = {
            'bbox': cat1_bboxes[idx],
            'filename': row.get('filename', ''),
            'area': row['bbox_w'] * row['bbox_h']
        }
    
    # Calculate containments for each category 0 instance
    containment_results = []
    containment_counts = []
    
    print(f"Analyzing containment with {overlap_threshold*100:.1f}% overlap threshold...")
    
    for i, cat0_bbox in enumerate(cat0_bboxes):
        cat0_row = cat0_df.iloc[i]
        cat0_id = cat0_ids[i]
        cat0_area = cat0_bbox[2] * cat0_bbox[3]  # width * height
        contained_count = 0
        contained_cat1_ids = []
        
        for j, cat1_bbox in enumerate(cat1_bboxes):
            cat1_id = cat1_ids[j]
            
            if is_contained_by(cat1_bbox, cat0_bbox, overlap_threshold):
                contained_count += 1
                contained_cat1_ids.append(cat1_id)
        
        # Calculate area per contained instance
        area_per_contained = cat0_area / contained_count if contained_count > 0 else float('inf')
        
        containment_counts.append(contained_count)
        containment_results.append(ContainmentResult(
            cat0_id=cat0_id,
            cat0_area=cat0_area,
            contained_count=contained_count,
            contained_cat1_ids=contained_cat1_ids,
            area_per_contained=area_per_contained,
            is_oversized=False,  # Will be set later
            is_undersized=False,  # Will be set later
            filename=cat0_row.get('filename', ''),
            bbox=cat0_bbox
        ))
        
        if contained_count > 0:
            print(f"  Category 0 bbox {cat0_id} (area={cat0_area:.0f}) contains {contained_count} category 1 instances: {contained_cat1_ids}")
    
    # Analyze size proportionality
    containment_results = analyze_size_proportionality(containment_results)
    
    # Count how many category 0 instances contain 0, 1, 2, ... category 1 instances
    distribution = defaultdict(int)
    for count in containment_counts:
        distribution[count] += 1
    
    return dict(distribution), containment_results, cat1_data


def analyze_size_proportionality(results: List[ContainmentResult]) -> List[ContainmentResult]:
    """
    Analyze size proportionality and flag disproportionate instances.
    
    Args:
        results: List of containment results
        
    Returns:
        Updated list with proportionality flags set
    """
    if not results:
        return results
    
    # Calculate statistics for instances that contain at least one category 1
    containing_results = [r for r in results if r.contained_count > 0]
    
    if not containing_results:
        return results
    
    # Calculate area per contained instance statistics
    area_per_contained_values = [r.area_per_contained for r in containing_results]
    mean_area_per_contained = np.mean(area_per_contained_values)
    std_area_per_contained = np.std(area_per_contained_values)
    
    # Calculate area statistics
    areas = [r.cat0_area for r in results]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    
    print(f"\nSize Analysis Statistics:")
    print(f"  Mean area per contained instance: {mean_area_per_contained:.0f} ± {std_area_per_contained:.0f}")
    print(f"  Mean category 0 area: {mean_area:.0f} ± {std_area:.0f}")
    
    # Define thresholds for flagging (using 2 standard deviations)
    oversized_threshold = mean_area_per_contained + 2 * std_area_per_contained
    undersized_threshold = mean_area_per_contained - 2 * std_area_per_contained
    
    # Also consider absolute area outliers
    large_area_threshold = mean_area + 2 * std_area
    small_area_threshold = mean_area - 2 * std_area
    
    # Update results with flags
    updated_results = []
    oversized_count = 0
    undersized_count = 0
    
    for result in results:
        is_oversized = False
        is_undersized = False
        
        if result.contained_count > 0:
            # Flag if area per contained instance is disproportionate
            if result.area_per_contained > oversized_threshold:
                is_oversized = True
                oversized_count += 1
            elif result.area_per_contained < undersized_threshold and result.area_per_contained > 0:
                is_undersized = True
                undersized_count += 1
        else:
            # For instances containing no category 1, flag if they're unusually large
            if result.cat0_area > large_area_threshold:
                is_oversized = True
                oversized_count += 1
        
        updated_results.append(ContainmentResult(
            cat0_id=result.cat0_id,
            cat0_area=result.cat0_area,
            contained_count=result.contained_count,
            contained_cat1_ids=result.contained_cat1_ids,
            area_per_contained=result.area_per_contained,
            is_oversized=is_oversized,
            is_undersized=is_undersized,
            filename=result.filename,
            bbox=result.bbox
        ))
    
    print(f"  Flagged {oversized_count} oversized instances")
    print(f"  Flagged {undersized_count} undersized instances")
    
    return updated_results


def create_debug_visualizations(containment_results: List[ContainmentResult], 
                               cat1_data: Dict, 
                               image_dir: str, 
                               output_dir: str, 
                               filter_count: Optional[int] = None):
    """
    Create debug visualizations for containment analysis.
    
    Args:
        containment_results: List of containment analysis results
        cat1_data: Dictionary mapping cat1 IDs to their data
        image_dir: Directory containing the original images
        output_dir: Directory to save visualization outputs
        filter_count: If specified, only visualize instances with this containment count
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Install opencv-python and matplotlib.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group results by containment count
    grouped_results = defaultdict(list)
    for result in containment_results:
        if filter_count is None or result.contained_count == filter_count:
            grouped_results[result.contained_count].append(result)
    
    # Also group disproportionate instances
    oversized_results = [r for r in containment_results if r.is_oversized]
    undersized_results = [r for r in containment_results if r.is_undersized]
    
    if filter_count is not None:
        oversized_results = [r for r in oversized_results if r.contained_count == filter_count]
        undersized_results = [r for r in undersized_results if r.contained_count == filter_count]
    
    print(f"\nCreating debug visualizations in: {output_dir}")
    
    # Create visualizations for each containment count
    for count, results in grouped_results.items():
        if not results:
            continue
            
        print(f"  Creating visualizations for containment count {count} ({len(results)} instances)")
        create_containment_visualization(results, cat1_data, image_dir, output_dir, count)
    
    # Create visualizations for disproportionate instances
    if oversized_results:
        print(f"  Creating visualizations for {len(oversized_results)} oversized instances")
        create_disproportionate_visualization(oversized_results, cat1_data, image_dir, output_dir, "oversized")
    
    if undersized_results:
        print(f"  Creating visualizations for {len(undersized_results)} undersized instances")
        create_disproportionate_visualization(undersized_results, cat1_data, image_dir, output_dir, "undersized")
    
    # Create summary visualization
    create_summary_visualization(containment_results, output_dir)


def create_containment_visualization(results: List[ContainmentResult], 
                                   cat1_data: Dict, 
                                   image_dir: str, 
                                   output_dir: str, 
                                   count: int):
    """Create visualization for instances with specific containment count."""
    # Limit to first 12 instances for readability
    results_to_show = results[:12]
    
    # Calculate grid size
    n_images = len(results_to_show)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1) if n_images > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Category 0 Instances Containing {count} Category 1 Instance{"s" if count != 1 else ""}', 
                 fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results_to_show):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load and display image
        image_path = os.path.join(image_dir, result.filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                # Draw category 0 bounding box (parent) in red
                x, y, w, h = result.bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', 
                                       facecolor='none', linestyle='-', label='Category 0')
                ax.add_patch(rect)
                
                # Draw category 1 bounding boxes (children) in blue
                for cat1_id in result.contained_cat1_ids:
                    if cat1_id in cat1_data:
                        x1, y1, w1, h1 = cat1_data[cat1_id]['bbox']
                        rect = patches.Rectangle((x1, y1), w1, h1, linewidth=1, edgecolor='blue', 
                                               facecolor='none', linestyle='--', label='Category 1')
                        ax.add_patch(rect)
                
                # Add title with details
                title = f'ID: {result.cat0_id}\nArea: {result.cat0_area:.0f}'
                if result.contained_count > 0:
                    title += f'\nArea/Count: {result.area_per_contained:.0f}'
                
                # Add flags
                flags = []
                if result.is_oversized:
                    flags.append('OVERSIZED')
                if result.is_undersized:
                    flags.append('UNDERSIZED')
                if flags:
                    title += f'\n{", ".join(flags)}'
                
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Image not found\n{result.filename}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Image not found\n{result.filename}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    # Add legend
    if n_images > 0:
        red_patch = patches.Patch(color='red', label='Category 0 (Parent)')
        blue_patch = patches.Patch(color='blue', label='Category 1 (Child)')
        fig.legend(handles=[red_patch, blue_patch], loc='lower center', ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'containment_count_{count}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_disproportionate_visualization(results: List[ContainmentResult], 
                                        cat1_data: Dict, 
                                        image_dir: str, 
                                        output_dir: str, 
                                        flag_type: str):
    """Create visualization for disproportionate instances."""
    # Limit to first 12 instances for readability
    results_to_show = results[:12]
    
    # Calculate grid size
    n_images = len(results_to_show)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1) if n_images > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'{flag_type.capitalize()} Instances', fontsize=16, fontweight='bold')
    
    for idx, result in enumerate(results_to_show):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Load and display image
        image_path = os.path.join(image_dir, result.filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                
                # Draw category 0 bounding box in orange for flagged instances
                x, y, w, h = result.bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='orange', 
                                       facecolor='none', linestyle='-')
                ax.add_patch(rect)
                
                # Draw category 1 bounding boxes in blue
                for cat1_id in result.contained_cat1_ids:
                    if cat1_id in cat1_data:
                        x1, y1, w1, h1 = cat1_data[cat1_id]['bbox']
                        rect = patches.Rectangle((x1, y1), w1, h1, linewidth=1, edgecolor='blue', 
                                               facecolor='none', linestyle='--')
                        ax.add_patch(rect)
                
                # Add title with details
                title = f'ID: {result.cat0_id} ({flag_type.upper()})\nArea: {result.cat0_area:.0f}'
                title += f'\nCount: {result.contained_count}'
                if result.contained_count > 0:
                    title += f'\nArea/Count: {result.area_per_contained:.0f}'
                
                ax.set_title(title, fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Image not found\n{result.filename}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{flag_type}_instances.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_visualization(results: List[ContainmentResult], output_dir: str):
    """Create summary statistics visualization."""
    # Count distribution
    counts = [r.contained_count for r in results]
    areas = [r.cat0_area for r in results]
    area_per_contained = [r.area_per_contained for r in results if r.contained_count > 0]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Containment count distribution
    unique_counts, count_freq = np.unique(counts, return_counts=True)
    ax1.bar(unique_counts, count_freq, alpha=0.7)
    ax1.set_xlabel('Number of Contained Category 1 Instances')
    ax1.set_ylabel('Number of Category 0 Instances')
    ax1.set_title('Containment Count Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Area distribution
    ax2.hist(areas, bins=30, alpha=0.7)
    ax2.set_xlabel('Category 0 Area (pixels)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Category 0 Area Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Area per contained instance
    if area_per_contained:
        ax3.hist(area_per_contained, bins=30, alpha=0.7)
        ax3.set_xlabel('Area per Contained Instance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Area per Contained Instance Distribution')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No instances with\ncontained category 1', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Area per Contained Instance Distribution')
    
    # Scatter plot: area vs contained count
    ax4.scatter(counts, areas, alpha=0.6)
    ax4.set_xlabel('Number of Contained Category 1 Instances')
    ax4.set_ylabel('Category 0 Area (pixels)')
    ax4.set_title('Area vs Containment Count')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def print_distribution_statistics(distribution: Dict[int, int]):
    """Print the distribution statistics in a readable format."""
    total_cat0 = sum(distribution.values())
    
    print(f"\nCategory Containment Distribution:")
    print(f"Total category 0 instances analyzed: {total_cat0}")
    print(f"\nDistribution of category 1 instances contained within category 0:")
    
    for cat1_count in sorted(distribution.keys()):
        cat0_count = distribution[cat1_count]
        percentage = (cat0_count / total_cat0) * 100 if total_cat0 > 0 else 0
        
        if cat1_count == 0:
            print(f"  {cat0_count:3d} category 0 instances contain  0 category 1 instances ({percentage:5.1f}%)")
        elif cat1_count == 1:
            print(f"  {cat0_count:3d} category 0 instances contain  1 category 1 instance  ({percentage:5.1f}%)")
        else:
            print(f"  {cat0_count:3d} category 0 instances contain {cat1_count:2d} category 1 instances ({percentage:5.1f}%)")
    
    # Summary statistics
    max_contained = max(distribution.keys()) if distribution else 0
    avg_contained = sum(k * v for k, v in distribution.items()) / total_cat0 if total_cat0 > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Maximum category 1 instances in a single category 0: {max_contained}")
    print(f"  Average category 1 instances per category 0: {avg_contained:.2f}")


def print_disproportionate_instances(results: List[ContainmentResult]):
    """Print details about disproportionate instances."""
    oversized = [r for r in results if r.is_oversized]
    undersized = [r for r in results if r.is_undersized]
    
    if oversized:
        print(f"\nOversized Instances (disproportionately large for their content):")
        print(f"{'ID':>6} {'Area':>8} {'Count':>5} {'Area/Count':>10} {'Contained IDs'}")
        print("-" * 60)
        for r in sorted(oversized, key=lambda x: x.area_per_contained, reverse=True):
            area_per_str = f"{r.area_per_contained:.0f}" if r.area_per_contained != float('inf') else "∞"
            contained_str = str(r.contained_cat1_ids) if len(r.contained_cat1_ids) <= 5 else f"{r.contained_cat1_ids[:5]}..."
            print(f"{r.cat0_id:>6} {r.cat0_area:>8.0f} {r.contained_count:>5} {area_per_str:>10} {contained_str}")
    
    if undersized:
        print(f"\nUndersized Instances (disproportionately small for their content):")
        print(f"{'ID':>6} {'Area':>8} {'Count':>5} {'Area/Count':>10} {'Contained IDs'}")
        print("-" * 60)
        for r in sorted(undersized, key=lambda x: x.area_per_contained):
            contained_str = str(r.contained_cat1_ids) if len(r.contained_cat1_ids) <= 5 else f"{r.contained_cat1_ids[:5]}..."
            print(f"{r.cat0_id:>6} {r.cat0_area:>8.0f} {r.contained_count:>5} {r.area_per_contained:>10.0f} {contained_str}")
    
    if not oversized and not undersized:
        print(f"\nNo disproportionate instances detected.")


def main():
    parser = argparse.ArgumentParser(description='Analyze category containment and create debug visualizations')
    parser.add_argument('metadata_path', help='Path to metadata CSV file')
    parser.add_argument('--overlap-threshold', type=float, default=0.7, 
                       help='Overlap threshold for containment (default: 0.7)')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create debug visualizations')
    parser.add_argument('--image-dir', type=str, 
                       help='Directory containing original images (required for visualization)')
    parser.add_argument('--output-dir', type=str, default='debug_visualizations',
                       help='Output directory for visualizations (default: debug_visualizations)')
    parser.add_argument('--filter-count', type=int,
                       help='Only visualize instances with specific containment count')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.metadata_path):
        print(f"Error: Input file '{args.metadata_path}' not found.")
        sys.exit(1)
    
    # Validate visualization requirements
    if args.visualize:
        if not VISUALIZATION_AVAILABLE:
            print("Error: Visualization requires opencv-python and matplotlib. Install with:")
            print("pip install opencv-python matplotlib")
            sys.exit(1)
        
        if not args.image_dir:
            print("Error: --image-dir is required for visualization")
            sys.exit(1)
        
        if not os.path.exists(args.image_dir):
            print(f"Error: Image directory '{args.image_dir}' not found.")
            sys.exit(1)
    
    print(f"Loading metadata from: {args.metadata_path}")
    print(f"Overlap threshold: {args.overlap_threshold*100:.1f}%")
    
    try:
        # Load the metadata CSV
        df = pd.read_csv(args.metadata_path)
        
        # Validate required columns
        required_cols = ['bbox_x0', 'bbox_y0', 'bbox_w', 'bbox_h', 'category']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        print(f"Loaded {len(df)} total instances from metadata file.")
        
        # Analyze category containment
        distribution, containment_results, cat1_data = analyze_category_containment(df, args.overlap_threshold)
        
        # Print results
        print_distribution_statistics(distribution)
        print_disproportionate_instances(containment_results)
        
        # Create visualizations if requested
        if args.visualize:
            create_debug_visualizations(containment_results, cat1_data, args.image_dir, 
                                      args.output_dir, args.filter_count)
            print(f"\nDebug visualizations saved to: {args.output_dir}")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
