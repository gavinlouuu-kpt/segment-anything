#!/usr/bin/env python3
"""
Example script showing how to use the pipeline with example data.

This script demonstrates running the pipeline with the same parameters
that were shown in the original pipeline.py example workflow.
"""

import os
import subprocess
import sys


def main():
    # Example parameters from the original pipeline.py
    example_input = "input/PC3D2/test_00101_Cam_V710_Cine1_png.rf.8a4931e50b7f32883a623779ece85187.jpg"
    example_output_base = "output"  # Pipeline will create unique subfolder
    default_checkpoint = "models/sam_vit_h_4b8939.pth"
    
    # Check if input file exists
    if not os.path.exists(example_input):
        print(f"Example input file not found: {example_input}")
        print("Please ensure you have the example data or modify the paths in this script.")
        sys.exit(1)
    
    # Check if default checkpoint exists
    if not os.path.exists(default_checkpoint):
        print(f"Default SAM checkpoint not found: {default_checkpoint}")
        print("Please download the SAM checkpoint:")
        print(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P models/")
        sys.exit(1)
    
    # Build the pipeline command (checkpoint will use default)
    cmd = [
        "python", "scripts/pipeline.py",
        "--input", example_input,
        "--output", example_output_base,
        "--model-type", "vit_h",
        "--device", "cuda",
        "--crop-n-layers", "3",
        "--overlap-threshold", "0.8",
        "--circularity-threshold", "0.6",
        "--visualize",
        "--filter-count", "1",
        "--debug"
    ]
    
    print("Running pipeline with example parameters...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\nExample pipeline completed successfully!")
        print(f"Results saved to unique folder in: {example_output_base}")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with return code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main() 