#!/usr/bin/env python3
"""
Mask Area Statistics and Visualization Script

This script analyzes mask directories and creates visualizations of mask area distributions.
It can handle both individual mask PNG files and metadata CSV files from SAM outputs.

Usage:
    python scripts/statistics.py <mask_directory> [--output_dir output_plots]

Example:
    python scripts/statistics.py output/fluor_bead
    python scripts/statistics.py output/fluor_bead --output_dir analysis_results
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Optional
import json


def load_masks_from_csv(mask_dir: str) -> pd.DataFrame:
    """
    Load mask metadata from CSV file in the mask directory.
    
    Args:
        mask_dir: Path to directory containing mask files and metadata.csv
        
    Returns:
        DataFrame with mask metadata including areas
    """
    csv_path = Path(mask_dir) / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metadata.csv found in {mask_dir}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} mask records from metadata.csv")
    return df


def load_masks_from_images(mask_dir: str) -> pd.DataFrame:
    """
    Load mask information by analyzing PNG files directly.
    
    Args:
        mask_dir: Path to directory containing mask PNG files
        
    Returns:
        DataFrame with computed mask areas
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory {mask_dir} does not exist")
    
    mask_files = sorted(list(mask_dir.glob("*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No PNG files found in {mask_dir}")
    
    data = []
    print(f"Analyzing {len(mask_files)} mask images...")
    
    for i, mask_file in enumerate(mask_files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(mask_files)} masks")
            
        # Load mask image
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Calculate area (number of non-zero pixels)
            area = np.sum(mask > 128)  # Threshold at 128 for binary masks
            
            data.append({
                'id': i,
                'filename': mask_file.name,
                'area': area,
                'height': mask.shape[0],
                'width': mask.shape[1]
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Analyzed {len(df)} mask images")
    return df


def create_area_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive statistics for mask areas.
    
    Args:
        df: DataFrame with mask data including 'area' column
        
    Returns:
        Dictionary with statistical measures
    """
    areas = df['area'].values
    
    stats = {
        'count': len(areas),
        'mean': np.mean(areas),
        'median': np.median(areas),
        'std': np.std(areas),
        'min': np.min(areas),
        'max': np.max(areas),
        'q1': np.percentile(areas, 25),
        'q3': np.percentile(areas, 75),
        'iqr': np.percentile(areas, 75) - np.percentile(areas, 25),
        'skewness': pd.Series(areas).skew(),
        'kurtosis': pd.Series(areas).kurtosis()
    }
    
    # Add percentiles
    for p in [5, 10, 90, 95, 99]:
        stats[f'p{p}'] = np.percentile(areas, p)
    
    return stats


def create_visualizations(df: pd.DataFrame, output_dir: str, mask_dir: str):
    """
    Create comprehensive visualizations of mask area distributions.
    
    Args:
        df: DataFrame with mask data
        output_dir: Directory to save plots
        mask_dir: Original mask directory name for titles
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate statistics
    stats = create_area_statistics(df)
    areas = df['area'].values
    
    # 1. Main distribution plot with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Mask Area Distribution Analysis - {Path(mask_dir).name}', fontsize=16, fontweight='bold')
    
    # Determine optimal bin count for main histograms - increased for better detail
    hist_bins = min(150, max(50, len(areas) // 8))
    
    # Histogram
    axes[0,0].hist(areas, bins=hist_bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.0f}')
    axes[0,0].axvline(stats['median'], color='green', linestyle='--', label=f'Median: {stats["median"]:.0f}')
    axes[0,0].set_xlabel('Area (pixels)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Area Distribution (Linear Scale)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Log-scale histogram
    axes[0,1].hist(areas, bins=hist_bins, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_yscale('log')
    axes[0,1].set_xlabel('Area (pixels)')
    axes[0,1].set_ylabel('Frequency (log scale)')
    axes[0,1].set_title('Area Distribution (Log Scale)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Box plot
    box_plot = axes[1,0].boxplot(areas, patch_artist=True, tick_labels=['Mask Areas'])
    box_plot['boxes'][0].set_facecolor('lightgreen')
    axes[1,0].set_ylabel('Area (pixels)')
    axes[1,0].set_title('Area Distribution (Box Plot)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_areas = np.sort(areas)
    cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    axes[1,1].plot(sorted_areas, cumulative, linewidth=2, color='purple')
    axes[1,1].set_xlabel('Area (pixels)')
    axes[1,1].set_ylabel('Cumulative Probability')
    axes[1,1].set_title('Cumulative Distribution Function')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'mask_area_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed frequency analysis
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Detailed Frequency Analysis - {Path(mask_dir).name}', fontsize=14, fontweight='bold')
    
    # Create bins for frequency analysis - use many more bins for better detail
    # Use a more sophisticated binning strategy with much higher resolution
    area_range = areas.max() - areas.min()
    if area_range > 10000:
        n_bins = min(500, max(100, len(areas) // 5))  # Much more bins for large ranges
    elif area_range > 1000:
        n_bins = min(300, max(80, len(areas) // 8))
    else:
        n_bins = min(200, max(50, len(areas) // 12))
    
    print(f"  Using {n_bins} bins for frequency analysis (range: {area_range:.0f} pixels)")
    counts, bin_edges = np.histogram(areas, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Frequency histogram with percentile markers
    # Limit x-axis to 95th percentile for better visualization
    p95 = np.percentile(areas, 95)
    p99 = np.percentile(areas, 99)
    
    axes[0].bar(bin_centers, counts, width=np.diff(bin_edges), alpha=0.7, color='steelblue', edgecolor='white')
    
    # Add percentile lines
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    colors = ['orange', 'red', 'green', 'red', 'orange', 'purple', 'black']
    for p, color in zip(percentiles, colors):
        pval = np.percentile(areas, p)
        axes[0].axvline(pval, color=color, linestyle='--', alpha=0.8, 
                       label=f'P{p}: {pval:.0f}')
    
    # Limit x-axis to show meaningful data (95th percentile + some margin)
    x_limit = min(p95 * 1.2, p99)
    axes[0].set_xlim(0, x_limit)
    
    # Add info about truncated data
    outliers_count = len(areas[areas > x_limit])
    outlier_percentage = (outliers_count / len(areas)) * 100
    
    axes[0].set_xlabel('Area (pixels)')
    axes[0].set_ylabel('Frequency')
    title = f'Frequency Distribution with Percentiles (showing up to {x_limit:.0f} pixels)'
    if outliers_count > 0:
        title += f'\n({outliers_count} outliers > {x_limit:.0f} pixels not shown, {outlier_percentage:.1f}%)'
    axes[0].set_title(title)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Area ranges analysis
    ranges = [
        (0, 100, 'Very Small'),
        (100, 500, 'Small'), 
        (500, 1000, 'Small-Medium'),
        (1000, 2000, 'Medium'),
        (2000, 5000, 'Medium-Large'),
        (5000, 10000, 'Large'),
        (10000, float('inf'), 'Very Large')
    ]
    
    range_counts = []
    range_labels = []
    for min_area, max_area, label in ranges:
        count = len(df[(df['area'] >= min_area) & (df['area'] < max_area)])
        if count > 0:
            range_counts.append(count)
            range_labels.append(f'{label}\n({min_area}-{max_area if max_area != float("inf") else "∞"})')
    
    # Pie chart of area ranges
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(range_counts)))
    wedges, texts, autotexts = axes[1].pie(range_counts, labels=range_labels, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90)
    axes[1].set_title('Distribution by Area Ranges')
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Statistical summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a text summary of statistics
    summary_text = f"""
MASK AREA STATISTICS SUMMARY
{'='*40}

Dataset: {Path(mask_dir).name}
Total Masks: {stats['count']:,}

CENTRAL TENDENCY:
Mean Area: {stats['mean']:.1f} pixels
Median Area: {stats['median']:.1f} pixels

SPREAD:
Standard Deviation: {stats['std']:.1f} pixels
Interquartile Range: {stats['iqr']:.1f} pixels
Range: {stats['min']:.0f} - {stats['max']:.0f} pixels

PERCENTILES:
5th: {stats['p5']:.0f}    25th: {stats['q1']:.0f}    50th: {stats['median']:.0f}
75th: {stats['q3']:.0f}   95th: {stats['p95']:.0f}   99th: {stats['p99']:.0f}

DISTRIBUTION SHAPE:
Skewness: {stats['skewness']:.3f} ({'right-skewed' if stats['skewness'] > 0.5 else 'left-skewed' if stats['skewness'] < -0.5 else 'approximately symmetric'})
Kurtosis: {stats['kurtosis']:.3f} ({'heavy-tailed' if stats['kurtosis'] > 0 else 'light-tailed'})

AREA RANGES:
Very Small (0-100): {len(df[df['area'] < 100]):,} masks ({len(df[df['area'] < 100])/len(df)*100:.1f}%)
Small (100-500): {len(df[(df['area'] >= 100) & (df['area'] < 500)]):,} masks ({len(df[(df['area'] >= 100) & (df['area'] < 500)])/len(df)*100:.1f}%)
Medium (500-2000): {len(df[(df['area'] >= 500) & (df['area'] < 2000)]):,} masks ({len(df[(df['area'] >= 500) & (df['area'] < 2000)])/len(df)*100:.1f}%)
Large (2000-5000): {len(df[(df['area'] >= 2000) & (df['area'] < 5000)]):,} masks ({len(df[(df['area'] >= 2000) & (df['area'] < 5000)])/len(df)*100:.1f}%)
Very Large (5000+): {len(df[df['area'] >= 5000]):,} masks ({len(df[df['area'] >= 5000])/len(df)*100:.1f}%)
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Statistical Summary - {Path(mask_dir).name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / 'statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics as JSON (convert numpy types to native Python types)
    stats_json = {k: float(v) if hasattr(v, 'item') else int(v) if isinstance(v, (np.integer, np.int64)) else v 
                  for k, v in stats.items()}
    with open(output_path / 'statistics.json', 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    print(f"✓ Visualizations saved to {output_path}")
    print(f"  - mask_area_distribution.png")
    print(f"  - frequency_analysis.png") 
    print(f"  - statistical_summary.png")
    print(f"  - statistics.json")


def print_summary(df: pd.DataFrame, mask_dir: str):
    """Print a quick summary to console."""
    stats = create_area_statistics(df)
    
    print(f"\n{'='*60}")
    print(f"MASK AREA ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {Path(mask_dir).name}")
    print(f"Total masks: {stats['count']:,}")
    print(f"Mean area: {stats['mean']:.1f} pixels")
    print(f"Median area: {stats['median']:.1f} pixels")
    print(f"Area range: {stats['min']:.0f} - {stats['max']:.0f} pixels")
    print(f"Standard deviation: {stats['std']:.1f} pixels")
    print(f"{'='*60}")


def main():
    """Main function to run the mask statistics analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze mask area distributions and create visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/statistics.py output/fluor_bead
  python scripts/statistics.py output/fluor_bead --output_dir analysis_results
  python scripts/statistics.py path/to/masks --output_dir plots
        """
    )
    
    parser.add_argument('mask_dir', 
                       type=str,
                       help='Path to directory containing mask files')
    
    parser.add_argument('--output_dir', 
                       type=str,
                       default=None,
                       help='Directory to save analysis plots and results (default: auto-generated unique folder)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.mask_dir):
        print(f"❌ Error: Directory '{args.mask_dir}' does not exist")
        sys.exit(1)
    
    # Generate unique output directory if not specified
    if args.output_dir is None:
        mask_dir_name = Path(args.mask_dir).name
        args.output_dir = f"mask_analysis/{mask_dir_name}"
    
    print(f"🔍 Analyzing masks in: {args.mask_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    try:
        # Try to load from CSV first (faster and more complete data)
        try:
            df = load_masks_from_csv(args.mask_dir)
            data_source = "metadata.csv"
        except FileNotFoundError:
            # Fall back to analyzing PNG files directly
            print("📝 No metadata.csv found, analyzing PNG files directly...")
            df = load_masks_from_images(args.mask_dir)
            data_source = "PNG analysis"
        
        print(f"📈 Data loaded from: {data_source}")
        
        # Print console summary
        print_summary(df, args.mask_dir)
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        create_visualizations(df, args.output_dir, args.mask_dir)
        
        print(f"\n✅ Analysis complete! Check '{args.output_dir}' for results.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
