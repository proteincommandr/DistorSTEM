#!/usr/bin/env python3
import numpy as np
import mrcfile
import argparse
import os
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
    Apply anisotropic scaling to the image while maintaining original dimensions.
    The content is warped but the output size remains the same.
    Args:
        image: Input image as numpy array
        x_scale: Scaling factor in x direction
        y_scale: Scaling factor in y direction
    Returns:
        Distorted image with same dimensions as input
    """
    height, width = image.shape
    y, x = np.mgrid[0:height, 0:width]
    
    # Create coordinate mapping that scales from the center
    center_y, center_x = height // 2, width // 2
    
    # Scale coordinates around the center
    x = center_x + (x - center_x) / x_scale
    y = center_y + (y - center_y) / y_scale
    
    # Map the coordinates
    coords = np.array([y, x])
    
    # Apply the transformation while maintaining original dimensions
    return ndimage.map_coordinates(image, coords, order=1)

def apply_logarithmic_distortion(image, amplitude=10, decay=0.5, quarter_width=None):
    """
    Apply logarithmic distortion to the left quarter of the image.
    The distortion is centered vertically around the middle of the image.
    Args:
        image: Input image as numpy array
        amplitude: Maximum displacement in pixels
        decay: Rate of decay for the logarithmic function
        quarter_width: Width of affected region (default: image_width/4)
    Returns:
        Distorted image with same dimensions as input
    """
    height, width = image.shape
    if quarter_width is None:
        quarter_width = width // 4
    
    # Create coordinate grid
    y, x = np.mgrid[0:height, 0:width]
    center_y = height // 2
    
    # Create displacement field for left quarter
    x_coord = np.arange(quarter_width)
    displacement = amplitude * np.exp(-decay * x_coord)
    
    # Apply displacement relative to the center line
    y_offset = np.zeros_like(image)
    y_shifted = y - center_y
    y_offset[:, :quarter_width] = np.tile(displacement, (height, 1))
    # Scale displacement based on distance from center
    y_offset = y_offset * (y_shifted / center_y)
    
    # Create transformed coordinates
    coords = np.array([y + y_offset, x])
    
    # Apply transformation
    return ndimage.map_coordinates(image, coords, order=1)

def apply_y_scaling(image, scale_factor=2.0):
    """
    Apply progressive y-direction scaling from left to right.
    Args:
        image: Input image as numpy array
        scale_factor: Maximum distortion in pixels. The right edge of the image
                     will be shifted by this amount relative to the center line.
                     Positive values inflate the right side, negative values compress it.
    Returns:
        Distorted image with same dimensions as input
    """
    height, width = image.shape
    y, x = np.mgrid[0:height, 0:width]
    center = height // 2
    
    # Create scaling that goes from 1 at left to (1 + scale_factor/center) at right
    scale = 1.0 + (scale_factor / center) * (x / (width - 1))
    # Apply scaling relative to center line
    y_shifted = y - center
    y_scaled = y_shifted * scale + center

def generate_output_filename(input_path, args):
    """
    Generate output filename with descriptive suffix based on applied distortions.
    """
    base = os.path.splitext(input_path)[0]
    suffix = []
    
    if args.x_scale != 1.0 or args.y_scale != 1.0:
        suffix.append(f"aniso_x{args.x_scale:.2f}_y{args.y_scale:.2f}")
    if args.log_amplitude > 0:
        suffix.append(f"log_a{args.log_amplitude:.1f}_d{args.log_decay:.1f}")
    if args.y_scale_factor > 0:
        suffix.append(f"yscale_{args.y_scale_factor:.4f}")
    
    if not suffix:
        return f"{base}_nodistort.mrc"
    return f"{base}_{'_'.join(suffix)}.mrc"

def process_mrc_file(input_path, args):
    """
    Process a single MRC file with the specified distortions.
    """
    output_path = generate_output_filename(input_path, args)
    print(f"Processing {input_path} -> {os.path.basename(output_path)}")
    
    with mrcfile.open(input_path) as mrc:
        image = mrc.data.copy()
    
    # Apply selected distortions only
    if args.x_scale != 1.0 or args.y_scale != 1.0:
        image = apply_anisotropy(image, args.x_scale, args.y_scale)
    if args.log_amplitude > 0:
        image = apply_logarithmic_distortion(image, args.log_amplitude, args.log_decay)
    if args.y_scale_factor > 0:
        image = apply_y_scaling(image, args.y_scale_factor)
    
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(image.astype(np.float32))

def main():
    parser = argparse.ArgumentParser(description='Create and manipulate STEM distortion patterns')
    parser.add_argument('--output_grid', type=str, help='Output path for grid image')
    parser.add_argument('--input_dir', type=str, help='Input directory containing MRC files to process')
    parser.add_argument('--x_scale', type=float, default=1.0, help='X direction scaling factor (unitless)')
    parser.add_argument('--y_scale', type=float, default=1.0, help='Y direction scaling factor (unitless)')
    parser.add_argument('--log_amplitude', type=float, default=0, help='Maximum displacement for logarithmic distortion (pixels)')
    parser.add_argument('--log_decay', type=float, default=0.5, help='Decay rate of logarithmic distortion')
    parser.add_argument('--y_scale_factor', type=float, default=0, help='Progressive Y scaling in pixels (positive: inflate right side, negative: compress)')
    
    args = parser.parse_args()
    
    # Create grid image
    if args.output_grid:
        grid = create_grid_image()
        with mrcfile.new(args.output_grid, overwrite=True) as mrc:
            mrc.set_data(grid)
    
    # Process input directory if provided
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: {args.input_dir} is not a directory")
            return
        
        mrc_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mrc')]
        if not mrc_files:
            print(f"No MRC files found in {args.input_dir}")
            return
        
        print(f"Found {len(mrc_files)} MRC files to process")
        for mrc_file in mrc_files:
            input_path = os.path.join(args.input_dir, mrc_file)
            process_mrc_file(input_path, args)

if __name__ == '__main__':
    main()