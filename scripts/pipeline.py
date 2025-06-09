#!/usr/bin/env python3
"""
Segment Anything Mask Processing Pipeline

This script orchestrates the complete pipeline for mask generation, filtering, categorization, and analysis.
It runs the following steps in sequence:
1. amg.py - Generate masks using SAM
2. filter.py - Filter masks based on circularity and overlap
3. categorise.py - Categorize masks as parent/child based on containment
4. category_counting.py - Analyze distribution and optionally visualize

Usage:
    python scripts/pipeline.py --input <input_image_or_dir> --output <output_dir> [options]
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List
import json


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def validate_paths(input_path: str, output_path: str) -> tuple[str, str]:
    """Validate and normalize input and output paths."""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return input_path, output_path


def get_image_filename(input_path: str) -> str:
    """Extract image filename from input path for directory naming."""
    if os.path.isfile(input_path):
        return Path(input_path).stem
    else:
        # For directories, use the directory name
        return Path(input_path).name


def step1_generate_masks(input_path: str, output_path: str, checkpoint: str, 
                        model_type: str, device: str, crop_n_layers: int) -> tuple[bool, str]:
    """Step 1: Generate masks using AMG."""
    amg_output = os.path.join(output_path, "amg_output")
    
    cmd = [
        "python", "scripts/amg.py",
        "--input", input_path,
        "--output", amg_output,
        "--checkpoint", checkpoint,
        "--model-type", model_type,
        "--device", device,
        "--crop-n-layers", str(crop_n_layers)
    ]
    
    success = run_command(cmd, "Step 1: Generating masks with SAM")
    
    # Find the generated metadata.csv file
    metadata_file = None
    if success:
        if os.path.isfile(input_path):
            # Single image - metadata should be in a subdirectory named after the image
            image_name = Path(input_path).stem
            metadata_file = os.path.join(amg_output, image_name, "metadata.csv")
        else:
            # Directory - need to find the metadata file
            for root, dirs, files in os.walk(amg_output):
                if "metadata.csv" in files:
                    metadata_file = os.path.join(root, "metadata.csv")
                    break
        
        if metadata_file and os.path.exists(metadata_file):
            print(f"Generated metadata file: {metadata_file}")
        else:
            print("Warning: Could not find generated metadata.csv file")
            success = False
    
    return success, metadata_file or ""


def step2_filter_masks(metadata_file: str, output_path: str, overlap_threshold: float, 
                      circularity_threshold: float, debug: bool) -> tuple[bool, str]:
    """Step 2: Filter masks based on circularity and overlap."""
    input_dir = os.path.dirname(metadata_file)
    filter_output = os.path.join(output_path, "filtered")
    
    cmd = [
        "python", "scripts/filter.py",
        input_dir,
        "--output_dir", filter_output,
        "--overlap_threshold", str(overlap_threshold),
        "--circularity_threshold", str(circularity_threshold)
    ]
    
    if debug:
        cmd.append("--debug")
    
    success = run_command(cmd, "Step 2: Filtering masks")
    
    filtered_metadata = os.path.join(filter_output, "metadata.csv")
    if success and os.path.exists(filtered_metadata):
        print(f"Generated filtered metadata: {filtered_metadata}")
    else:
        print("Warning: Could not find filtered metadata.csv file")
        success = False
    
    return success, filtered_metadata if success else ""


def step3_categorize_masks(metadata_file: str, output_path: str, 
                          overlap_threshold: float, debug: bool) -> tuple[bool, str]:
    """Step 3: Categorize masks as parent/child."""
    categorized_file = os.path.join(output_path, "metadata_categorized.csv")
    
    cmd = [
        "python", "scripts/categorise.py",
        metadata_file,
        categorized_file,
        str(overlap_threshold)
    ]
    
    if debug:
        cmd.append("--debug")
    
    success = run_command(cmd, "Step 3: Categorizing masks")
    
    if success and os.path.exists(categorized_file):
        print(f"Generated categorized metadata: {categorized_file}")
    else:
        print("Warning: Could not find categorized metadata file")
        success = False
    
    return success, categorized_file if success else ""


def step4_analyze_categories(metadata_file: str, output_path: str, overlap_threshold: float,
                           visualize: bool, image_dir: Optional[str], filter_count: Optional[int]) -> bool:
    """Step 4: Analyze category distribution."""
    cmd = [
        "python", "scripts/category_counting.py",
        metadata_file,
        "--overlap-threshold", str(overlap_threshold)
    ]
    
    if visualize:
        cmd.append("--visualize")
        if image_dir:
            cmd.extend(["--image-dir", image_dir])
        visualization_output = os.path.join(output_path, "visualizations")
        cmd.extend(["--output-dir", visualization_output])
        
        if filter_count is not None:
            cmd.extend(["--filter-count", str(filter_count)])
    
    success = run_command(cmd, "Step 4: Analyzing category distribution")
    return success


def create_pipeline_summary(output_path: str, input_path: str, args: argparse.Namespace):
    """Create a summary file with pipeline configuration and results."""
    summary = {
        "pipeline_config": {
            "input_path": input_path,
            "output_path": output_path,
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "device": args.device,
            "crop_n_layers": args.crop_n_layers,
            "overlap_threshold": args.overlap_threshold,
            "circularity_threshold": args.circularity_threshold,
            "debug": args.debug,
            "visualize": args.visualize,
            "filter_count": args.filter_count
        },
        "output_files": {
            "amg_output": os.path.join(output_path, "amg_output"),
            "filtered_output": os.path.join(output_path, "filtered"),
            "categorized_metadata": os.path.join(output_path, "metadata_categorized.csv"),
            "visualizations": os.path.join(output_path, "visualizations") if args.visualize else None
        }
    }
    
    summary_file = os.path.join(output_path, "pipeline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPipeline summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Segment Anything Mask Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single image
  python scripts/pipeline.py --input image.jpg --output results --checkpoint models/sam_vit_h_4b8939.pth

  # With visualization and debug mode
  python scripts/pipeline.py --input image.jpg --output results --checkpoint models/sam_vit_h_4b8939.pth --visualize --debug

  # Process directory of images
  python scripts/pipeline.py --input input_dir --output results --checkpoint models/sam_vit_h_4b8939.pth
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory of images")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to SAM checkpoint file")
    
    # SAM model arguments
    parser.add_argument("--model-type", type=str, default="vit_h",
                       choices=["default", "vit_h", "vit_l", "vit_b"],
                       help="SAM model type (default: vit_h)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (default: cuda)")
    parser.add_argument("--crop-n-layers", type=int, default=3,
                       help="Number of crop layers for AMG (default: 3)")
    
    # Filtering arguments
    parser.add_argument("--overlap-threshold", type=float, default=0.7,
                       help="Overlap threshold for filtering and categorization (default: 0.7)")
    parser.add_argument("--circularity-threshold", type=float, default=0.6,
                       help="Circularity threshold for filtering (default: 0.6)")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true",
                       help="Generate debug visualizations")
    parser.add_argument("--filter-count", type=int,
                       help="Only visualize instances with this many contained objects")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode for detailed logging")
    
    # Validation mode
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate arguments and show what would be run without executing")
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        input_path, output_path = validate_paths(args.input, args.output)
        
        print("=== Segment Anything Mask Processing Pipeline ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Model: {args.model_type} ({args.checkpoint})")
        print(f"Device: {args.device}")
        
        if args.dry_run:
            print("\n=== DRY RUN MODE - No commands will be executed ===")
            return
        
        # Step 1: Generate masks
        success, metadata_file = step1_generate_masks(
            input_path, output_path, args.checkpoint, args.model_type, 
            args.device, args.crop_n_layers
        )
        if not success:
            print("ERROR: Mask generation failed")
            sys.exit(1)
        
        # Step 2: Filter masks
        success, filtered_metadata = step2_filter_masks(
            metadata_file, output_path, args.overlap_threshold, 
            args.circularity_threshold, args.debug
        )
        if not success:
            print("ERROR: Mask filtering failed")
            sys.exit(1)
        
        # Step 3: Categorize masks
        success, categorized_metadata = step3_categorize_masks(
            filtered_metadata, output_path, args.overlap_threshold, args.debug
        )
        if not success:
            print("ERROR: Mask categorization failed")
            sys.exit(1)
        
        # Step 4: Analyze categories
        image_dir = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        success = step4_analyze_categories(
            categorized_metadata, output_path, args.overlap_threshold,
            args.visualize, image_dir if args.visualize else None, args.filter_count
        )
        if not success:
            print("ERROR: Category analysis failed")
            sys.exit(1)
        
        # Create summary
        create_pipeline_summary(output_path, input_path, args)
        
        print("\n=== Pipeline completed successfully! ===")
        print(f"Results saved to: {output_path}")
        
        if args.visualize:
            print(f"Visualizations saved to: {os.path.join(output_path, 'visualizations')}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()