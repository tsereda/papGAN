#!/usr/bin/env python3
"""
FID Score Calculator using TorchMetrics

This script calculates the Fréchet Inception Distance (FID) between sets of images.
It compares generated images against real healthy and unhealthy image sets.
Results are saved to a file.
"""
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import os
import argparse
import sys
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import datetime
import random

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

def calculate_fid(path1, path2, batch_size=32, device='cuda', feature_dim=2048, validation=False, val_samples=50):
    """Calculate FID between two image directories using TorchMetrics."""
    if not (check_dir(path1) and check_dir(path2)):
        print(f"Error: One or both directories do not exist!")
        return None
    
    try:
        print(f"\nCalculating FID between {path1} and {path2}...")
        
        # Initialize FID metric
        fid = FrechetInceptionDistance(feature=feature_dim).to(device)
        
        # Get lists of image files
        path1_files = [os.path.join(path1, f) for f in os.listdir(path1) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        path2_files = [os.path.join(path2, f) for f in os.listdir(path2) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Sample a subset of images if in validation mode
        if validation:
            print(f"VALIDATION MODE: Sampling {val_samples} images from each directory")
            if len(path1_files) > val_samples:
                path1_files = random.sample(path1_files, val_samples)
            if len(path2_files) > val_samples:
                path2_files = random.sample(path2_files, val_samples)
            print(f"Using {len(path1_files)} images from directory 1 and {len(path2_files)} images from directory 2")
        
        # Image transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 input size
            transforms.ToTensor(),
        ])
        
        # Process images from path1 (real) in batches
        print(f"Processing images from {path1}...")
        for i in tqdm(range(0, len(path1_files), batch_size)):
            batch_files = path1_files[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if batch_images:
                # Stack batch and convert to uint8 [0, 255]
                batch_tensor = torch.stack(batch_images)
                batch_tensor = (batch_tensor * 255).to(torch.uint8).to(device)
                fid.update(batch_tensor, real=True)
        
        # Process images from path2 (generated) in batches
        print(f"Processing images from {path2}...")
        for i in tqdm(range(0, len(path2_files), batch_size)):
            batch_files = path2_files[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if batch_images:
                # Stack batch and convert to uint8 [0, 255]
                batch_tensor = torch.stack(batch_images)
                batch_tensor = (batch_tensor * 255).to(torch.uint8).to(device)
                fid.update(batch_tensor, real=False)
        
        # Compute FID score
        print("Computing FID score...")
        score = fid.compute().item()
        print(f"FID score: {score:.4f}")
        return score
        
    except Exception as e:
        print(f"Error calculating FID: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results_to_file(output_file, healthy_path, unhealthy_path, generated_path, 
                        fid_vs_healthy, fid_vs_unhealthy, args):
    """Save FID results to a file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w') as f:
        f.write(f"=== FID Score Results ===\n")
        f.write(f"Date and Time: {timestamp}\n\n")
        
        f.write("=== Settings ===\n")
        f.write(f"Real healthy images: {healthy_path}\n")
        f.write(f"Real unhealthy images: {unhealthy_path}\n")
        f.write(f"Generated images: {generated_path}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Feature dimensions: {args.dims}\n")
        f.write(f"Validation mode: {args.validation}\n")
        if args.validation:
            f.write(f"Validation samples: {args.val_samples}\n\n")
        else:
            f.write("\n")
        
        f.write("=== Results ===\n")
        if fid_vs_healthy is not None:
            f.write(f"FID vs healthy (A): {fid_vs_healthy:.4f}\n")
        else:
            f.write("FID vs healthy (A): Failed to calculate\n")
        
        if fid_vs_unhealthy is not None:
            f.write(f"FID vs unhealthy (B): {fid_vs_unhealthy:.4f}\n\n")
        else:
            f.write("FID vs unhealthy (B): Failed to calculate\n\n")
        
        # Analysis
        if fid_vs_healthy is not None and fid_vs_unhealthy is not None:
            f.write("=== Analysis ===\n")
            if fid_vs_healthy < fid_vs_unhealthy:
                f.write(f"The generated images are more similar to healthy images (A) by {fid_vs_unhealthy - fid_vs_healthy:.4f} FID points\n")
            elif fid_vs_unhealthy < fid_vs_healthy:
                f.write(f"The generated images are more similar to unhealthy images (B) by {fid_vs_healthy - fid_vs_unhealthy:.4f} FID points\n")
            else:
                f.write("The generated images are equally similar to both healthy and unhealthy images\n")
    
    print(f"\nResults saved to: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate FID scores between image sets")
    
    parser.add_argument("--real-healthy", type=str, required=True,
                        help="Path to directory with real healthy images")
    
    parser.add_argument("--real-unhealthy", type=str, required=True,
                        help="Path to directory with real unhealthy images")
    
    parser.add_argument("--generated", type=str, required=True,
                        help="Path to directory with generated images")
    
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for FID calculation (default: 32)")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for calculation: 'cuda' or 'cpu' (default: cuda)")
    
    parser.add_argument("--dims", type=int, default=2048,
                        help="Dimensionality of Inception features (default: 2048)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    parser.add_argument("--output", type=str, default="fid_results.txt",
                        help="Output file to save results (default: fid_results.txt)")
    
    parser.add_argument("--validation", action="store_true",
                        help="Run in validation mode with fewer samples for faster execution")
    
    parser.add_argument("--val-samples", type=int, default=50,
                        help="Number of images to sample in validation mode (default: 50)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"
        else:
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Report settings
    print("\n=== FID Score Calculator (TorchMetrics) ===")
    print(f"Real healthy images: {args.real_healthy}")
    print(f"Real unhealthy images: {args.real_unhealthy}")
    print(f"Generated images: {args.generated}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Feature dimensions: {args.dims}")
    print(f"Output file: {args.output}")
    
    if args.validation:
        print(f"\n*** RUNNING IN VALIDATION MODE ***")
        print(f"Using only {args.val_samples} images from each directory for quick testing")
    
    # Calculate FID vs real healthy images (A)
    fid_vs_healthy = calculate_fid(
        args.real_healthy, 
        args.generated,
        batch_size=args.batch_size,
        device=args.device,
        feature_dim=args.dims,
        validation=args.validation,
        val_samples=args.val_samples
    )
    
    # Calculate FID vs real unhealthy images (B)
    fid_vs_unhealthy = calculate_fid(
        args.real_unhealthy, 
        args.generated,
        batch_size=args.batch_size,
        device=args.device,
        feature_dim=args.dims,
        validation=args.validation,
        val_samples=args.val_samples
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
    
    # Save results to file
    save_results_to_file(
        args.output,
        args.real_healthy,
        args.real_unhealthy,
        args.generated,
        fid_vs_healthy,
        fid_vs_unhealthy,
        args
    )

if __name__ == "__main__":
    main()