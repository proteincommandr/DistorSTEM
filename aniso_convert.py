#!/usr/bin/env python3
"""
Convert CryoSPARC anisotropic magnification matrices to scaling matrices.

This script provides utilities to:
1. Reconstruct the true 2x2 anisotropic magnification matrix from CryoSPARC's stored format
2. Decompose the matrix into rotation and scaling components using SVD
3. Extract rotation angles and scale factors for physical interpretation
"""

import numpy as np
import argparse
from pathlib import Path


def rotation_angle(R):
    """
    Calculate rotation angle from a 2x2 rotation matrix.
    
    Args:
        R: 2x2 rotation matrix
    
    Returns:
        float: Rotation angle in degrees
    """
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def analyze_aniso_matrix(A_stored):
    """
    Analyze a CryoSPARC anisotropic magnification matrix.
    
    Args:
        A_stored: 2x2 stored matrix from CryoSPARC dataset
        
    Returns:
        dict: Analysis results including rotations, scales, and matrices
    """
    # Reconstruct true magnification matrix
    A = np.eye(2) + A_stored
    
    # Perform singular value decomposition
    U, S, Vt = np.linalg.svd(A)
    
    # Extract rotation angles
    theta_U = rotation_angle(U)
    theta_V = rotation_angle(Vt.T)
    
    # Create result dictionary
    results = {
        'matrix_stored': A_stored,
        'matrix_full': A,
        'U': U,
        'S': S,
        'V': Vt.T,
        'rotation_pre': theta_V,
        'rotation_post': theta_U,
        'scale_factors': S,
        'reconstruction_error': np.linalg.norm(A - U @ np.diag(S) @ Vt)
    }
    
    return results


def print_analysis(results):
    """
    Print formatted analysis results.
    
    Args:
        results: Dictionary of analysis results from analyze_aniso_matrix
    """
    print("\nCryoSPARC Anisotropic Magnification Analysis")
    print("=" * 45)
    
    print("\nStored matrix (CryoSPARC format):")
    print(results['matrix_stored'])
    
    print("\nReconstructed anisotropic magnification matrix:")
    print(results['matrix_full'])
    
    print("\nDecomposition (A = U @ S @ V^T):")
    print("U (post-rotation):")
    print(results['U'])
    print("\nSingular values (scale factors):")
    print(results['S'])
    print("\nV (pre-rotation):")
    print(results['V'])
    
    print("\nRotation angles:")
    print(f"Pre-scaling rotation  (V): {results['rotation_pre']:.6f}°")
    print(f"Post-scaling rotation (U): {results['rotation_post']:.6f}°")
    
    print("\nScale factors:")
    print(f"Major axis: {results['scale_factors'][0]:.6f}")
    print(f"Minor axis: {results['scale_factors'][1]:.6f}")
    print(f"Aspect ratio: {results['scale_factors'][0]/results['scale_factors'][1]:.6f}")
    
    print(f"\nReconstruction error: {results['reconstruction_error']:.2e}")
    
    print("\nDirect input for stem_distortion.py:")
    print(f"--x_scale {results['scale_factors'][0]:.6f} --y_scale {results['scale_factors'][1]:.6f}")


def main():
    """Main function to demonstrate usage with example matrix."""
    # Example matrix from CryoSPARC
    A_stored = np.array([
        [2.41974842e-03, 1.46644180e-05],
        [1.51588251e-04, -5.78628697e-04]
    ])
    
    # Analyze the matrix
    results = analyze_aniso_matrix(A_stored)
    
    # Print results
    print_analysis(results)


if __name__ == "__main__":
    main()