#!/usr/bin/env python3
"""
Visualize CycleGAN results by creating side-by-side comparisons
of healthy inputs and their corresponding unhealthy outputs.
"""
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import random

def create_comparison_grid(input_dir, generated_dir, output_path, num_examples=5):
    """
    Create a grid of side-by-side comparisons between input and generated images.
    
    Args:
        input_dir: Directory containing input (healthy) images
        generated_dir: Directory containing generated (unhealthy) images
        output_path: Path to save the comparison image
        num_examples: Number of examples to show
    """
    print(f"Creating comparison grid with {num_examples} examples...")
    
    # Get all input files and their corresponding generated files
    input_files = list(Path(input_dir).glob('real_healthy_*.png'))
    
    # Ensure we have enough images
    if len(input_files) < num_examples:
        print(f"Warning: Only {len(input_files)} images available. Showing all.")
        num_examples = len(input_files)
    
    # Randomly select examples if we have more than we need
    if len(input_files) > num_examples:
        input_files = random.sample(input_files, num_examples)
    
    # Sort the selected files to ensure consistent ordering
    input_files.sort()
    
    # Prepare figure
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3*num_examples))
    
    # If only one example, wrap axes in a list
    if num_examples == 1:
        axes = [axes]
    
    # For each input file, find the corresponding generated file
    for i, input_file in enumerate(input_files):
        # Extract index from filename (e.g., real_healthy_0001.png -> 0001)
        idx = input_file.stem.split('_')[-1]
        
        # Construct path to corresponding generated file
        generated_file = Path(generated_dir) / f"fake_unhealthy_{idx}.png"
        
        if not generated_file.exists():
            print(f"Warning: Generated file {generated_file} not found. Skipping.")
            continue
        
        # Load images
        input_img = np.array(Image.open(input_file))
        generated_img = np.array(Image.open(generated_file))
        
        # Display images
        axes[i][0].imshow(input_img)
        axes[i][0].set_title(f"Input (Healthy) #{idx}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(generated_img)
        axes[i][1].set_title(f"Generated (Unhealthy) #{idx}")
        axes[i][1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    print(f"Saving comparison to {output_path}")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    print("Comparison created successfully!")

def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparisons of CycleGAN results")
    parser.add_argument('--input_dir', type=str, default='results_20k/unhealthy/test_latest/images/input',
                       help='Directory containing input (healthy) images')
    parser.add_argument('--generated_dir', type=str, default='results_20k/unhealthy/test_latest/images/generated',
                       help='Directory containing generated (unhealthy) images')
    parser.add_argument('--output_path', type=str, default='comparison_results.png',
                       help='Path to save the comparison image')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to include in the comparison')
    
    args = parser.parse_args()
    
    # Create comparison
    create_comparison_grid(
        args.input_dir,
        args.generated_dir,
        args.output_path,
        args.num_examples
    )

if __name__ == "__main__":
    main()