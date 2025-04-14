#!/usr/bin/env python3
"""
FID Score Calculator

This script calculates the Fréchet Inception Distance (FID) between sets of images.
It compares generated images against real healthy and unhealthy image sets.
"""
import torch
import sys
print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available in script: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
from pytorch_fid import fid_score
import os
import argparse
import sys

def check_dir(path):
    """Check if directory exists and count files."""
    print(f"\nChecking {path}:")
    exists = os.path.exists(path)
    print(f"Exists: {exists}")
    
    if exists:
        files = os.listdir(path)
        num_files = len(files)
        print(f"Number of files: {num_files}")
        if num_files > 0:
            print(f"Sample filenames: {files[:3]}...")
    else:
        print("Directory not found!")
    
    return exists

def calculate_fid(path1, path2, batch_size=1, device='cuda', dims=2048):
    """Calculate FID between two image directories."""
    if not (check_dir(path1) and check_dir(path2)):
        print(f"Error: One or both directories do not exist!")
        return None
    
    try:
        print(f"\nCalculating FID between {path1} and {path2}...")
        fid = fid_score.calculate_fid_given_paths(
            [path1, path2],
            batch_size=batch_size,
            device=device,
            dims=dims
        )
        return fid
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate FID scores between image sets")
    
    parser.add_argument("--real-healthy", type=str, required=True,
                        help="Path to directory with real healthy images")
    
    parser.add_argument("--real-unhealthy", type=str, required=True,
                        help="Path to directory with real unhealthy images")
    
    parser.add_argument("--generated", type=str, required=True,
                        help="Path to directory with generated images")
    
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for FID calculation (default: 1)")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for calculation: 'cuda' or 'cpu' (default: cuda)")
    
    parser.add_argument("--dims", type=int, default=2048,
                        help="Dimensionality of Inception features (default: 2048)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
# Check if CUDA is available if requested
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
        else:
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Report settings
    print("\n=== FID Score Calculator ===")
    print(f"Real healthy images: {args.real_healthy}")
    print(f"Real unhealthy images: {args.real_unhealthy}")
    print(f"Generated images: {args.generated}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Feature dimensions: {args.dims}")
    
    # Calculate FID vs real healthy images (A)
    fid_vs_healthy = calculate_fid(
        args.real_healthy, 
        args.generated,
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims
    )
    
    # Calculate FID vs real unhealthy images (B)
    fid_vs_unhealthy = calculate_fid(
        args.real_unhealthy, 
        args.generated,
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims
    )
    
    # Print results
    print("\n=== Results ===")
    if fid_vs_healthy is not None:
        print(f"FID vs healthy (A): {fid_vs_healthy:.4f}")
    
    if fid_vs_unhealthy is not None:
        print(f"FID vs unhealthy (B): {fid_vs_unhealthy:.4f}")
    
    # Compare the scores
    if fid_vs_healthy is not None and fid_vs_unhealthy is not None:
        print("\n=== Analysis ===")
        if fid_vs_healthy < fid_vs_unhealthy:
            print(f"The generated images are more similar to healthy images (A) by {fid_vs_unhealthy - fid_vs_healthy:.4f} FID points")
        elif fid_vs_unhealthy < fid_vs_healthy:
            print(f"The generated images are more similar to unhealthy images (B) by {fid_vs_healthy - fid_vs_unhealthy:.4f} FID points")
        else:
            print("The generated images are equally similar to both healthy and unhealthy images")

if __name__ == "__main__":
    main()