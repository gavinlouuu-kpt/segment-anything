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
from typing import List, Tuple, Optional
import random


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_masks_from_directory(mask_dir: str) -> List[np.ndarray]:
    """Load all mask images from a directory."""
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        raise ValueError(f"Mask directory {mask_dir} does not exist")
    
    masks = []
    mask_files = sorted([f for f in mask_dir.glob("*.png") if f.name != "metadata.csv"])
    
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Convert to binary mask (0 or 1)
            mask = (mask > 128).astype(np.uint8)
            masks.append(mask)
    
    return masks


def generate_random_colors(num_colors: int) -> List[Tuple[float, float, float]]:
    """Generate random colors for mask visualization."""
    colors = []
    for _ in range(num_colors):
        colors.append((random.random(), random.random(), random.random()))
    return colors


def create_overlay_image(image: np.ndarray, masks: List[np.ndarray], 
                        alpha: float = 0.5, colors: Optional[List[Tuple[float, float, float]]] = None) -> np.ndarray:
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
    return np.clip(result, 0, 255).astype(np.uint8)


def create_side_by_side_visualization(image: np.ndarray, masks: List[np.ndarray], 
                                    colors: Optional[List[Tuple[float, float, float]]] = None) -> plt.Figure:
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
    overlay = create_overlay_image(image, masks, colors=colors)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Overlay ({len(masks)} masks)")
    axes[0, 1].axis('off')
    
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


class InteractiveMaskViewer:
    """Interactive viewer for examining masks overlayed on an image."""
    
    def __init__(self, image: np.ndarray, masks: List[np.ndarray], colors: List[Tuple[float, float, float]]):
        self.image = image
        self.masks = masks
        self.colors = colors
        self.current_mask_idx = 0
        self.show_mask = True
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
            overlay = create_overlay_image(self.image, [current_mask], 
                                         self.alpha, [self.colors[self.current_mask_idx]])
            self.ax.imshow(overlay)
            title = f"Mask {self.current_mask_idx + 1}/{len(self.masks)} (Alpha: {self.alpha:.1f})"
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
            
        elif event.key == 'a':  # Show all masks
            self.ax.clear()
            overlay = create_overlay_image(self.image, self.masks, self.alpha, self.colors)
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


def start_interactive_mode(image: np.ndarray, masks: List[np.ndarray], colors: List[Tuple[float, float, float]]):
    """Start the interactive mask viewing mode."""
    if not masks:
        print("No masks available for interactive viewing!")
        return
    
    try:
        viewer = InteractiveMaskViewer(image, masks, colors)
        viewer.show()
    except ImportError:
        print("Interactive mode requires tkinter. Please install it or use non-interactive mode.")
    except Exception as e:
        print(f"Error starting interactive mode: {e}")
        print("Falling back to non-interactive visualization...")
        # Fallback to regular visualization
        fig = create_side_by_side_visualization(image, masks, colors)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay masks on an input image for visualization"
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
    
    args = parser.parse_args()
    
    try:
        # Load image and masks
        print(f"Loading image from {args.input_image}")
        image = load_image(args.input_image)
        
        print(f"Loading masks from {args.mask_dir}")
        masks = load_masks_from_directory(args.mask_dir)
        
        if not masks:
            print("No masks found in the specified directory!")
            return
        
        print(f"Found {len(masks)} masks")
        
        # Generate colors for masks
        colors = generate_random_colors(len(masks))
        
        # Interactive mode takes precedence
        if args.interactive and not args.no_display:
            start_interactive_mode(image, masks, colors)
            return 0
        
        if args.show_individual:
            # Create comprehensive visualization
            fig = create_side_by_side_visualization(image, masks, colors)
        else:
            # Create simple overlay
            overlay = create_overlay_image(image, masks, args.alpha, colors)
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(overlay)
            ax.set_title(f"Image with {len(masks)} mask overlays")
            ax.axis('off')
        
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
