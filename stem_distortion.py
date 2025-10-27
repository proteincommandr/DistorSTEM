#!/usr/bin/env python3
import numpy as np
import mrcfile
import argparse
from scipy import ndimage

def create_grid_image(size=4096, spacing=50):
    """
    Create a grid image with horizontal and vertical lines.
    Args:
        size: Size of the square image in pixels
        spacing: Number of pixels between grid lines
    Returns:
        numpy array of shape (size, size) with grid pattern
    """
    # Create black background
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add horizontal and vertical lines
    image[::spacing, :] = 1.0  # Horizontal lines
    image[:, ::spacing] = 1.0  # Vertical lines
    
    return image

def apply_anisotropy(image, x_scale=1.0, y_scale=1.0):
    """
    Apply anisotropic scaling to the image.
    Args:
        image: Input image as numpy array
        x_scale: Scaling factor in x direction
        y_scale: Scaling factor in y direction
    Returns:
        Distorted image
    """
    return ndimage.zoom(image, (y_scale, x_scale), order=1)

def apply_logarithmic_distortion(image, amplitude=10, decay=0.5, quarter_width=None):
    """
    Apply logarithmic distortion to the left quarter of the image.
    Args:
        image: Input image as numpy array
        amplitude: Maximum displacement in pixels
        decay: Rate of decay for the logarithmic function
        quarter_width: Width of affected region (default: image_width/4)
    Returns:
        Distorted image
    """
    height, width = image.shape
    if quarter_width is None:
        quarter_width = width // 4
    
    # Create coordinate grid
    y, x = np.mgrid[0:height, 0:width]
    
    # Create displacement field for left quarter
    x_coord = np.arange(quarter_width)
    displacement = amplitude * np.exp(-decay * x_coord)
    
    # Apply displacement only to left quarter
    y_offset = np.zeros_like(image)
    y_offset[:, :quarter_width] = np.tile(displacement, (height, 1))
    
    # Create transformed coordinates
    coords = np.array([y + y_offset, x])
    
    # Apply transformation
    return ndimage.map_coordinates(image, coords, order=1)

def apply_y_scaling(image, scale_factor=0.001):
    """
    Apply progressive y-direction scaling from left to right.
    Args:
        image: Input image as numpy array
        scale_factor: Factor controlling the amount of scaling
    Returns:
        Distorted image
    """
    height, width = image.shape
    y, x = np.mgrid[0:height, 0:width]
    center = height // 2
    # Limit the maximum distortion to a couple of pixels (e.g., 2 pixels)
    max_distortion = 2.0  # pixels
    # Scaling factor goes from 1 at left to (1 + max_distortion/center) at right
    scale = 1.0 + (max_distortion / center) * (x / (width - 1))
    # Reverse effect: right side inflated, left side compressed
    y_shifted = y - center
    y_scaled = y_shifted * scale + center
    coords = np.array([y_scaled, x])
    return ndimage.map_coordinates(image, coords, order=1)

def main():
    parser = argparse.ArgumentParser(description='Create and manipulate STEM distortion patterns')
    parser.add_argument('--output_grid', type=str, help='Output path for grid image')
    parser.add_argument('--input_mrc', type=str, help='Input MRC file to distort')
    parser.add_argument('--output_mrc', type=str, help='Output path for distorted MRC')
    parser.add_argument('--x_scale', type=float, default=1.0, help='X direction scaling factor')
    parser.add_argument('--y_scale', type=float, default=1.0, help='Y direction scaling factor')
    parser.add_argument('--log_amplitude', type=float, default=10, help='Amplitude of logarithmic distortion')
    parser.add_argument('--log_decay', type=float, default=0.5, help='Decay rate of logarithmic distortion')
    parser.add_argument('--y_scale_factor', type=float, default=0.001, help='Progressive Y scaling factor')
    
    args = parser.parse_args()
    
    # Create grid image
    if args.output_grid:
        grid = create_grid_image()
        with mrcfile.new(args.output_grid, overwrite=True) as mrc:
            mrc.set_data(grid)
    
    # Process input MRC if provided
    if args.input_mrc and args.output_mrc:
        with mrcfile.open(args.input_mrc) as mrc:
            image = mrc.data.copy()
        
            # Apply selected distortions only
            if args.x_scale != 1.0 or args.y_scale != 1.0:
                image = apply_anisotropy(image, args.x_scale, args.y_scale)
            if args.log_amplitude > 0:
                image = apply_logarithmic_distortion(image, args.log_amplitude, args.log_decay)
            if args.y_scale_factor > 0:
                image = apply_y_scaling(image, args.y_scale_factor)
        
        with mrcfile.new(args.output_mrc, overwrite=True) as mrc:
            mrc.set_data(image.astype(np.float32))

if __name__ == '__main__':
    main()