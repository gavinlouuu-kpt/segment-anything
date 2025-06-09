#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import pandas as pd


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_masks_from_directory(mask_dir: str, metadata_df: Optional[pd.DataFrame] = None) -> Tuple[List[np.ndarray], List[str]]:
    """Load mask images from a directory and return masks with their file paths.
    If metadata_df is provided and has a 'status' column, only load masks with 'PASSED' status."""
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise ValueError(f"Mask directory {mask_dir} does not exist")
    
    masks = []
    mask_files = []
    mask_file_paths = sorted([f for f in mask_dir.glob("*.png") if f.name != "metadata.csv"])
    
    # Create a set of passed filenames if metadata is available and has status column
    passed_filenames = None
    if metadata_df is not None and 'status' in metadata_df.columns:
        passed_filenames = set(metadata_df[metadata_df['status'] == 'PASSED']['filename'].tolist())
        print(f"Found {len(passed_filenames)} masks with PASSED status in metadata")
    
    for mask_file_path in mask_file_paths:
        filename = mask_file_path.name
        
        # Skip masks that don't have PASSED status if we're filtering
        if passed_filenames is not None and filename not in passed_filenames:
            continue
            
        mask = cv2.imread(str(mask_file_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Convert to binary mask (0 or 1)
            mask = (mask > 128).astype(np.uint8)
            masks.append(mask)
            mask_files.append(str(mask_file_path))
    
    return masks, mask_files


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata CSV file containing mask information."""
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file {metadata_path} does not exist")
    
    df = pd.read_csv(metadata_path)
    return df


def get_category_colors() -> Dict[str, Tuple[float, float, float]]:
    """Define consistent colors for different categories."""
    return {
        '0': (1.0, 0.0, 0.0),    # Red for category 0
        '1': (0.0, 0.0, 1.0),    # Blue for category 1
        'FAILED': (1.0, 1.0, 0.0)  # Yellow for failed masks
    }


def generate_category_colors(mask_files: List[str], metadata_df: pd.DataFrame) -> List[Tuple[float, float, float]]:
    """Generate colors based on mask categories from metadata."""
    category_colors = get_category_colors()
    colors = []
    
    # Create a mapping from filename to metadata
    filename_to_metadata = {}
    for _, row in metadata_df.iterrows():
        filename_to_metadata[row['filename']] = row
    
    for mask_file in mask_files:
        # Extract ID from mask filename (e.g., "104.png" -> "104.png")
        mask_filename = os.path.basename(mask_file)
        
        if mask_filename in filename_to_metadata:
            metadata = filename_to_metadata[mask_filename]
            
            # Determine category based on status and category column
            if metadata['status'].startswith('FAILED'):
                category = 'FAILED'
            else:
                category = str(metadata['category'])
            
            colors.append(category_colors.get(category, (0.5, 0.5, 0.5)))  # Default gray if unknown
        else:
            # If no metadata found, use gray
            colors.append((0.5, 0.5, 0.5))
    
    return colors


def generate_random_colors(num_colors: int) -> List[Tuple[float, float, float]]:
    """Generate random colors for mask visualization."""
    colors = []
    for _ in range(num_colors):
        colors.append((random.random(), random.random(), random.random()))
    return colors


def add_mask_labels(image: np.ndarray, masks: List[np.ndarray], mask_files: List[str], 
                   colors: List[Tuple[float, float, float]]) -> np.ndarray:
    """Add mask name labels to the image at the centroid of each mask."""
    result = image.copy()
    
    for i, (mask, mask_file) in enumerate(zip(masks, mask_files)):
        if mask.shape[:2] != image.shape[:2]:
            # Resize mask to match image dimensions
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Find mask centroid
        mask_coords = np.where(mask > 0)
        if len(mask_coords[0]) == 0:
            continue  # Skip empty masks
            
        centroid_y = int(np.mean(mask_coords[0]))
        centroid_x = int(np.mean(mask_coords[1]))
        
        # Get mask name from file path
        mask_name = os.path.splitext(os.path.basename(mask_file))[0]
        
        # Choose text color based on mask color (use contrasting color)
        mask_color = colors[i % len(colors)]
        # Use white text if mask is dark, black if mask is light
        brightness = sum(mask_color) / 3
        text_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)
        
        # Add text with background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(mask_name, font, font_scale, thickness)
        
        # Calculate text position (centered on centroid)
        text_x = max(0, min(centroid_x - text_width // 2, image.shape[1] - text_width))
        text_y = max(text_height, min(centroid_y + text_height // 2, image.shape[0] - baseline))
        
        # Draw background rectangle
        bg_color = tuple(int(255 * c) for c in mask_color)
        cv2.rectangle(result, 
                     (text_x - 2, text_y - text_height - 2), 
                     (text_x + text_width + 2, text_y + baseline + 2), 
                     bg_color, -1)
        
        # Draw text
        cv2.putText(result, mask_name, (text_x, text_y), font, font_scale, text_color, thickness)
    
    return result


def create_overlay_image(image: np.ndarray, masks: List[np.ndarray], 
                        alpha: float = 0.5, colors: Optional[List[Tuple[float, float, float]]] = None,
                        mask_files: Optional[List[str]] = None, show_labels: bool = False) -> np.ndarray:
    """Create an overlay image with masks on top of the original image."""
    if not masks:
        return image
    
    overlay = image.copy().astype(np.float32)
    
    if colors is None:
        colors = generate_random_colors(len(masks))
    
    # Create a combined mask overlay
    mask_overlay = np.zeros_like(image, dtype=np.float32)
    
    for i, mask in enumerate(masks):
        if mask.shape[:2] != image.shape[:2]:
            # Resize mask to match image dimensions
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        color = colors[i % len(colors)]
        
        # Apply color to mask regions
        for c in range(3):
            mask_overlay[:, :, c] += mask * color[c] * 255
    
    # Blend the original image with the mask overlay
    result = (1 - alpha) * overlay + alpha * mask_overlay
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Add mask labels if requested
    if show_labels and mask_files:
        result = add_mask_labels(result, masks, mask_files, colors)
    
    return result


def create_side_by_side_visualization(image: np.ndarray, masks: List[np.ndarray], 
                                    colors: Optional[List[Tuple[float, float, float]]] = None,
                                    mask_files: Optional[List[str]] = None, show_labels: bool = False) -> plt.Figure:
    """Create a side-by-side visualization showing original image, individual masks, and overlay."""
    if colors is None:
        colors = generate_random_colors(len(masks))
    
    # Calculate grid dimensions
    num_masks = len(masks)
    cols = min(4, num_masks + 2)  # Original, overlay, and up to 2 individual masks per row
    rows = max(1, (num_masks + 1) // (cols - 2) + 1)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Show original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Show overlay
    overlay = create_overlay_image(image, masks, colors=colors, mask_files=mask_files, show_labels=show_labels)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Overlay ({len(masks)} masks)")
    axes[0, 1].axis('off')
    
    # Add legend for category colors
    add_category_legend(axes[0, 1], colors)
    
    # Show individual masks
    for i, mask in enumerate(masks):
        row = (i + 2) // cols
        col = (i + 2) % cols
        
        if row < rows and col < cols:
            # Create colored mask visualization
            colored_mask = np.zeros_like(image)
            color = colors[i % len(colors)]
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c] * 255
            
            axes[row, col].imshow(colored_mask.astype(np.uint8))
            axes[row, col].set_title(f"Mask {i + 1}")
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(masks) + 2, rows * cols):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def add_category_legend(ax, colors: List[Tuple[float, float, float]]):
    """Add a legend showing category colors."""
    category_colors = get_category_colors()
    legend_elements = []
    
    # Check which categories are present in the colors
    unique_colors = set(colors)
    
    for color in unique_colors:
        if color == (1.0, 0.0, 0.0):
            legend_elements.append(patches.Patch(color=color, label='Category 0'))
        elif color == (0.0, 0.0, 1.0):
            legend_elements.append(patches.Patch(color=color, label='Category 1'))
        elif color == (1.0, 1.0, 0.0):
            legend_elements.append(patches.Patch(color=color, label='Failed'))
        elif color == (0.5, 0.5, 0.5):
            legend_elements.append(patches.Patch(color=color, label='Unknown'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))


class InteractiveMaskViewer:
    """Interactive viewer for examining masks overlayed on an image."""
    
    def __init__(self, image: np.ndarray, masks: List[np.ndarray], colors: List[Tuple[float, float, float]], 
                 mask_files: Optional[List[str]] = None):
        self.image = image
        self.masks = masks
        self.colors = colors
        self.mask_files = mask_files
        self.current_mask_idx = 0
        self.show_mask = True
        self.show_labels = False
        self.alpha = 0.5
        
        # Set up the interactive backend
        matplotlib.use('TkAgg')
        
        # Create the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial display
        self.update_display()
        
    def update_display(self):
        """Update the display with current mask settings."""
        self.ax.clear()
        
        if self.show_mask and self.masks:
            # Show overlay with current mask
            current_mask = self.masks[self.current_mask_idx]
            current_mask_file = [self.mask_files[self.current_mask_idx]] if self.mask_files else None
            overlay = create_overlay_image(self.image, [current_mask], 
                                         self.alpha, [self.colors[self.current_mask_idx]],
                                         current_mask_file, self.show_labels)
            self.ax.imshow(overlay)
            mask_name = os.path.splitext(os.path.basename(self.mask_files[self.current_mask_idx]))[0] if self.mask_files else str(self.current_mask_idx + 1)
            title = f"Mask {mask_name} ({self.current_mask_idx + 1}/{len(self.masks)}) (Alpha: {self.alpha:.1f})"
        else:
            # Show original image
            self.ax.imshow(self.image)
            title = f"Original Image ({len(self.masks)} masks available)"
        
        self.ax.set_title(title)
        self.ax.axis('off')
        
        # Add instructions
        instructions = [
            "Controls:",
            "← → : Previous/Next mask",
            "↑ ↓ : Increase/Decrease transparency", 
            "Space: Toggle mask on/off",
            "L: Toggle mask labels",
            "A: Show all masks overlayed",
            "R: Reset to original image",
            "S: Save current view",
            "Q: Quit"
        ]
        
        self.ax.text(0.02, 0.98, "\n".join(instructions), 
                    transform=self.ax.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="white", alpha=0.8))
        
        self.fig.canvas.draw()
    
    def on_key_press(self, event):
        """Handle keyboard input for navigation."""
        if event.key == 'left' and self.masks:
            self.current_mask_idx = (self.current_mask_idx - 1) % len(self.masks)
            self.show_mask = True
            self.update_display()
            
        elif event.key == 'right' and self.masks:
            self.current_mask_idx = (self.current_mask_idx + 1) % len(self.masks)
            self.show_mask = True
            self.update_display()
            
        elif event.key == 'up':
            self.alpha = min(1.0, self.alpha + 0.1)
            self.update_display()
            
        elif event.key == 'down':
            self.alpha = max(0.0, self.alpha - 0.1)
            self.update_display()
            
        elif event.key == ' ':  # Space bar
            self.show_mask = not self.show_mask
            self.update_display()
            
        elif event.key == 'l':  # Toggle labels
            self.show_labels = not self.show_labels
            self.update_display()
            
        elif event.key == 'a':  # Show all masks
            self.ax.clear()
            overlay = create_overlay_image(self.image, self.masks, self.alpha, self.colors, 
                                         self.mask_files, self.show_labels)
            self.ax.imshow(overlay)
            self.ax.set_title(f"All {len(self.masks)} masks overlayed (Alpha: {self.alpha:.1f})")
            self.ax.axis('off')
            self.fig.canvas.draw()
            
        elif event.key == 'r':  # Reset to original
            self.show_mask = False
            self.update_display()
            
        elif event.key == 's':  # Save current view
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interactive_mask_view_{timestamp}.png"
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Current view saved as {filename}")
            
        elif event.key == 'q':  # Quit
            plt.close(self.fig)
    
    def show(self):
        """Display the interactive viewer."""
        print("\nStarting interactive mask viewer...")
        print("Use keyboard controls to navigate through masks.")
        plt.show()


def start_interactive_mode(image: np.ndarray, masks: List[np.ndarray], colors: List[Tuple[float, float, float]], 
                          mask_files: Optional[List[str]] = None):
    """Start the interactive mask viewing mode."""
    if not masks:
        print("No masks available for interactive viewing!")
        return
    
    try:
        viewer = InteractiveMaskViewer(image, masks, colors, mask_files)
        viewer.show()
    except ImportError:
        print("Interactive mode requires tkinter. Please install it or use non-interactive mode.")
    except Exception as e:
        print(f"Error starting interactive mode: {e}")
        print("Falling back to non-interactive visualization...")
        # Fallback to regular visualization
        fig = create_side_by_side_visualization(image, masks, colors, mask_files)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay masks on an input image for visualization. "
                   "If metadata.csv is available, masks are colored by category: "
                   "Red (Category 0), Blue (Category 1), Yellow (Failed)."
    )
    
    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="Path to the input image"
    )
    
    parser.add_argument(
        "--mask-dir",
        type=str,
        required=True,
        help="Path to directory containing mask images (PNG files)"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to metadata CSV file (default: <mask-dir>/metadata.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output visualization (optional)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency of mask overlay (0.0 to 1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--show-individual",
        action="store_true",
        help="Show individual masks in addition to overlay"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true", 
        help="Start interactive mode to examine masks individually"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the visualization (only save to file)"
    )
    
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show mask names as labels on each mask (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load image
        print(f"Loading image from {args.input_image}")
        image = load_image(args.input_image)
        
        # Load metadata first to filter masks by status
        metadata_path = args.metadata if args.metadata else os.path.join(args.mask_dir, "metadata.csv")
        metadata_df = None
        if os.path.exists(metadata_path):
            print(f"Loading metadata from {metadata_path}")
            metadata_df = load_metadata(metadata_path)
            if 'status' in metadata_df.columns:
                total_masks = len(metadata_df)
                passed_masks = len(metadata_df[metadata_df['status'] == 'PASSED'])
                print(f"Metadata contains {total_masks} total masks, {passed_masks} with PASSED status")
                print("Will only load masks with PASSED status")
            else:
                print("Metadata found but no 'status' column - loading all masks")
        else:
            print("No metadata.csv found - loading all masks")
        
        # Load masks (filtered by PASSED status if metadata available)
        print(f"Loading masks from {args.mask_dir}")
        masks, mask_files = load_masks_from_directory(args.mask_dir, metadata_df)
        
        if not masks:
            print("No masks found in the specified directory!")
            return
        
        print(f"Loaded {len(masks)} masks")
        
        # Generate colors for masks based on categories
        if metadata_df is not None:
            colors = generate_category_colors(mask_files, metadata_df)
            print("Using category-based colors")
            
            # Print category summary
            category_counts = {}
            for color in colors:
                if color == (1.0, 0.0, 0.0):
                    category_counts['Category 0 (Red)'] = category_counts.get('Category 0 (Red)', 0) + 1
                elif color == (0.0, 0.0, 1.0):
                    category_counts['Category 1 (Blue)'] = category_counts.get('Category 1 (Blue)', 0) + 1
                elif color == (1.0, 1.0, 0.0):
                    category_counts['Failed (Yellow)'] = category_counts.get('Failed (Yellow)', 0) + 1
                else:
                    category_counts['Unknown (Gray)'] = category_counts.get('Unknown (Gray)', 0) + 1
            
            print("Category distribution:")
            for category, count in category_counts.items():
                print(f"  {category}: {count} masks")
        else:
            print("No metadata.csv found, using random colors")
            colors = generate_random_colors(len(masks))
        
        # Interactive mode takes precedence
        if args.interactive and not args.no_display:
            start_interactive_mode(image, masks, colors, mask_files)
            return 0
        
        if args.show_individual:
            # Create comprehensive visualization
            fig = create_side_by_side_visualization(image, masks, colors, mask_files, args.show_labels)
        else:
            # Create simple overlay
            overlay = create_overlay_image(image, masks, args.alpha, colors, mask_files, args.show_labels)
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(overlay)
            ax.set_title(f"Image with {len(masks)} mask overlays")
            ax.axis('off')
            
            # Add legend for category colors
            add_category_legend(ax, colors)
        
        # Save if output path specified
        if args.output:
            print(f"Saving visualization to {args.output}")
            fig.savefig(args.output, dpi=300, bbox_inches='tight')
            print(f"Visualization saved successfully!")
        
        # Always save a default output if no output specified
        if not args.output:
            default_output = "mask_overlay_visualization.png"
            print(f"No output path specified, saving to {default_output}")
            fig.savefig(default_output, dpi=300, bbox_inches='tight')
            print(f"Visualization saved successfully!")
        
        # Display unless disabled (but warn if no display available)
        if not args.no_display:
            try:
                plt.show()
            except Exception as e:
                print(f"Cannot display visualization (no interactive display available): {e}")
                print("The visualization has been saved to file instead.")
        
        plt.close(fig)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
