# Interactive Mask Overlay Viewer

This interactive UI provides a comprehensive way to visualize mask overlays with a three-column layout for better analysis and exploration of segmentation results.

## Features

- **Built-in File Selection**: Browse and select image files and mask directories directly in the UI
- **Left Column**: Original image and overlay visualization
- **Middle Column**: Scrollable list of all masks with detailed metrics
- **Right Column**: Individual mask display when selected
- **Interactive Controls**: Alpha slider for overlay transparency, menu bar with shortcuts
- **Detailed Metrics**: Area, bounding box, centroid, and color information for each mask
- **Keyboard Shortcuts**: Quick access to common functions

## Requirements

The interactive viewer uses Python's built-in `tkinter` library along with the existing dependencies:
- `tkinter` (usually included with Python)
- `matplotlib`
- `opencv-python`
- `numpy`

## Usage

### Launch with File Selection UI

Simply run the application and use the built-in file browser:

```bash
python interactive_overlay.py
```

### Launch with Pre-selected Files

You can optionally specify files via command line:

```bash
python interactive_overlay.py --input-image path/to/image.jpg --mask-dir path/to/masks/
```

### With Custom Alpha

```bash
python interactive_overlay.py --input-image path/to/image.jpg --mask-dir path/to/masks/ --alpha 0.7
```

### Using the Helper Script

For convenience, you can also use the helper script:

```bash
python run_interactive_overlay.py
```

The helper script will automatically look for example data if no paths are provided, or you can specify them manually.

## Interface Layout

### File Selection Panel (Top)
- **Browse Image**: Select input image file (JPG, PNG, BMP, TIFF supported)
- **Browse Directory**: Select directory containing mask PNG files
- **Load Data**: Load the selected files (enabled when both are selected)
- **Status**: Shows current loading status and file information

### Left Column: Original & Overlay
- **Top**: Original input image
- **Bottom**: All masks overlaid on the original image with transparency

### Middle Column: Mask Metrics (Scrollable)
- List of all detected masks
- Each mask shows:
  - **Area**: Number of pixels in the mask
  - **Bounding Box**: (x, y, width, height) coordinates
  - **Centroid**: Center point of the mask
  - **Color**: RGB values used for visualization
- Click "View Mask X" buttons to select individual masks

### Right Column: Selected Mask
- Displays the currently selected mask in its assigned color
- Updates when you click on mask buttons in the middle column

### Bottom Controls
- **Alpha Slider**: Adjust transparency of the overlay (0.0 = transparent, 1.0 = opaque)

### Menu Bar
- **File**: Open image, select directory, reload data, exit
- **View**: Reset alpha, show all masks info
- **Help**: About dialog

## Interaction

1. **Select Files**: Use "Browse Image" and "Browse Directory" buttons, then click "Load Data"
2. **View Individual Masks**: Click any "View Mask X" button in the middle column
3. **Adjust Transparency**: Use the alpha slider at the bottom
4. **Scroll Through Masks**: Use mouse wheel or scrollbar in the middle column
5. **Analyze Metrics**: Review detailed information for each mask
6. **Use Keyboard Shortcuts**: 
   - `Ctrl+O`: Open image
   - `Ctrl+D`: Select mask directory
   - `F5`: Reload data
   - `Ctrl+Q`: Exit

## File Structure

The mask directory should contain PNG files with binary masks (0 for background, 255 for foreground). The viewer will automatically:
- Load all PNG files from the directory
- Convert them to binary masks
- Generate random colors for visualization
- Calculate metrics for each mask

## Example

```bash
# Navigate to the scripts directory
cd segment-anything/scripts/

# Run with your data
python interactive_overlay.py \
    --input-image ../examples/truck.jpg \
    --mask-dir ../examples/masks/ \
    --alpha 0.6
```

## Tips

- **Large Number of Masks**: The middle column is scrollable to handle many masks
- **Mask Selection**: Selected masks are highlighted with a different button style
- **Real-time Updates**: Alpha changes update the overlay immediately
- **Detailed Analysis**: Use the metrics to understand mask properties like size and position

## Troubleshooting

1. **No Display**: Make sure you have a GUI environment (X11 forwarding if using SSH)
2. **Import Errors**: Ensure you're running from the `segment-anything/scripts/` directory
3. **No Masks Found**: Check that your mask directory contains PNG files
4. **Image Loading Issues**: Verify the image path and format (JPG, PNG supported)

This interactive viewer makes it easy to explore and analyze segmentation results with detailed metrics and intuitive navigation. 