import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import random
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import concurrent.futures
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for CycleGAN')
    parser.add_argument('--steps', type=str, default='all', 
                        help='Steps to run: all, validate, augment, resize, split')
    parser.add_argument('--source_dir', type=str, default='isbi2025-ps3c-train-dataset',
                        help='Source directory with raw data')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Target size for image resizing')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    return parser.parse_args()

def validate_create_combined(args):
    """Step 1: Validate dataset and create combined unhealthy folder using parallel processing"""
    print("\n--- Step 1: Validating dataset and creating combined unhealthy folder ---")
    
    # Define paths
    paths = [
        f'{args.source_dir}/healthy',
        f'{args.source_dir}/unhealthy',
        f'{args.source_dir}/bothcells',
        f'{args.source_dir}/rubbish'
    ]

    # Initialize stats dictionary
    stats = {}

    # Analyze each path
    for path in paths:
        print(f"\nChecking path: {path}")
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue

        files = list(Path(path).glob('*.jpg')) + list(Path(path).glob('*.png'))
        print(f"Found {len(files)} files in {path}")
        sizes = []

        for f in files[:100]:
            try:
                with Image.open(f) as img:
                    sizes.append(img.size)
            except Exception as e:
                print(f"Error processing {f}: {e}")

        if sizes:
            widths, heights = zip(*sizes)
            stats[path] = {
                'count': len(files),
                'avg_size': f"{np.mean(widths):.0f}x{np.mean(heights):.0f}",
                'sample_count': len(sizes),
                'files': files
            }

    # Print analysis results
    print("\nAnalysis Results:")
    for path, info in stats.items():
        print(f"\nFolder: {path}")
        print(f"Total images: {info['count']}")
        print(f"Average size: {info['avg_size']}")

    # Create combined folder for 'both' and 'unhealthy'
    combined_path = f'{args.source_dir}/both_and_unhealthy'
    os.makedirs(combined_path, exist_ok=True)

    # Helper function for copying a single file with source prefix
    def copy_file_with_prefix(file_info, dest_dir):
        source, file = file_info
        dest_file = os.path.join(dest_dir, f"{source}_{file.name}")
        shutil.copy2(file, dest_file)
        return dest_file
    
    # Prepare file list for parallel copying
    copy_tasks = []
    sources = ['bothcells', 'unhealthy']
    
    for source in sources:
        source_path = f'{args.source_dir}/{source}'
        print(f"\nPreparing copy tasks for: {source_path}")
        if source_path in stats:
            print(f"Number of files to copy: {len(stats[source_path]['files'])}")
            # Create (source_name, file_path) pairs
            for file in stats[source_path]['files']:
                copy_tasks.append((source, file))
    
    # Parallel copy
    print(f"\nStarting parallel file copy using process pool...")
    print(f"Total files to copy: {len(copy_tasks)}")
    
    # Set number of workers based on system
    max_workers = min(32, os.cpu_count() + 4)
    print(f"Using {max_workers} worker processes")
    
    copied_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use partial to fix the destination directory
        copy_fn = partial(copy_file_with_prefix, dest_dir=combined_path)
        
        # Execute copy operations in parallel with progress bar
        for _ in tqdm(executor.map(copy_fn, copy_tasks), total=len(copy_tasks)):
            copied_count += 1
    
    print(f"\nCreated combined folder: {combined_path}")
    print(f"Copied {copied_count} images using parallel processing")
    
    return stats

def augment_create_matched(args, stats=None):
    """Step 3: Augment data and create matched dataset with parallel processing"""
    print("\n--- Step 3: Augmenting data and creating matched dataset ---")
    
    # Setup paths
    healthy_dir = f'{args.source_dir}/healthy'
    combined_dir = f'{args.source_dir}/both_and_unhealthy'
    output_dir = Path('cyclegan_dataset_matched')

    trainA = output_dir / 'trainA'
    trainB = output_dir / 'trainB'
    test_healthy = output_dir / 'test_healthy'
    
    trainA.mkdir(parents=True, exist_ok=True)
    trainB.mkdir(parents=True, exist_ok=True)
    test_healthy.mkdir(parents=True, exist_ok=True)

    # Define augmentation transforms
    transform_list = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomRotation(90),
    ]

    # Get file lists
    healthy_files = list(Path(healthy_dir).glob('*.jpg')) + list(Path(healthy_dir).glob('*.png'))
    unhealthy_files = list(Path(combined_dir).glob('*.jpg')) + list(Path(combined_dir).glob('*.png'))

    # Parallel function to get image size
    def get_image_size(file_path):
        try:
            with Image.open(file_path) as img:
                return np.array(img.size)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    # Get target size from unhealthy images using parallel processing
    print('Analyzing unhealthy images in parallel...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        size_results = list(tqdm(
            executor.map(get_image_size, unhealthy_files),
            total=len(unhealthy_files)
        ))
    
    # Filter out None values from errors
    unhealthy_sizes = np.array([size for size in size_results if size is not None])
    target_size = unhealthy_sizes.mean(axis=0)
    print(f"Target size computed: {target_size[0]:.1f} x {target_size[1]:.1f}")

    # Function to copy and augment a single unhealthy image
    def process_unhealthy_image(args):
        src, counter, transform_list, dest_dir = args
        try:
            img = Image.open(src)
            # Copy original
            dest_file = dest_dir / f'unhealthy_{counter:04d}{src.suffix}'
            shutil.copy(src, dest_file)
            
            # Apply one random transform
            transform = random.choice(transform_list)
            aug_img = transform(img)
            aug_file = dest_dir / f'unhealthy_aug_{counter+1:04d}{src.suffix}'
            aug_img.save(aug_file)
            
            return [dest_file, aug_file]
        except Exception as e:
            print(f"Error processing {src}: {e}")
            return []

    # Prepare unhealthy processing tasks
    print('\nPreparing unhealthy image processing tasks...')
    unhealthy_tasks = [
        (src, i*2, transform_list, trainB) 
        for i, src in enumerate(unhealthy_files)
    ]
    
    # Process unhealthy images in parallel
    print('Processing and augmenting unhealthy images in parallel...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_unhealthy_image, unhealthy_tasks),
            total=len(unhealthy_tasks)
        ))
    
    # Flatten the results list and count files
    processed_files = [f for sublist in results for f in sublist if f]
    trainB_count = len(processed_files)
    print(f'\nTotal unhealthy images after augmentation: {trainB_count}')

    # Function to compute size match for a healthy image
    def compute_size_match(args):
        file_path, target = args
        try:
            with Image.open(file_path) as img:
                size = np.array(img.size)
                return (np.linalg.norm(size - target), file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return (float('inf'), file_path)  # Return infinity for error cases
    
    # Prepare healthy image matching tasks
    healthy_tasks = [(f, target_size) for f in healthy_files]
    
    # Process healthy images in parallel to find size matches
    print('\nSelecting size-matched healthy images in parallel...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        healthy_matches = list(tqdm(
            executor.map(compute_size_match, healthy_tasks),
            total=len(healthy_tasks)
        ))
    
    # Sort by size similarity and split into train and test
    print("Sorting results by size similarity...")
    sorted_healthy = sorted(healthy_matches, key=lambda x: x[0])
    selected_healthy = [f for _, f in sorted_healthy[:trainB_count]]
    test_healthy_files = [f for _, f in sorted_healthy[trainB_count:trainB_count+1000]]  # Limit test set size
    
    # Function to copy a file with indexed name
    def copy_with_index(args):
        src, index, prefix, dest_dir = args
        try:
            dest_file = dest_dir / f'{prefix}_{index:04d}{src.suffix}'
            shutil.copy(src, dest_file)
            return dest_file
        except Exception as e:
            print(f"Error copying {src}: {e}")
            return None
    
    # Prepare file copy tasks
    train_tasks = [(src, i, 'healthy', trainA) for i, src in enumerate(selected_healthy)]
    test_tasks = [(src, i, 'healthy', test_healthy) for i, src in enumerate(test_healthy_files)]
    
    # Copy train healthy files in parallel
    print('\nCopying training healthy files in parallel...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        train_results = list(tqdm(
            executor.map(copy_with_index, train_tasks),
            total=len(train_tasks)
        ))
    
    # Copy test healthy files in parallel
    print('\nCopying test healthy files in parallel...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        test_results = list(tqdm(
            executor.map(copy_with_index, test_tasks),
            total=len(test_tasks)
        ))
    
    # Count successful copies
    trainA_count = len([f for f in train_results if f])
    test_count = len([f for f in test_results if f])
    
    print(f'\nDataset created at {output_dir}:')
    print(f'trainA (healthy): {trainA_count} images')
    print(f'trainB (unhealthy+bothcells): {trainB_count} images')
    print(f'test_healthy: {test_count} images')

    # Verify sample sizes
    print("\nSample image sizes:")
    for folder, name in [(trainA, 'trainA'), (trainB, 'trainB'), (test_healthy, 'test_healthy')]:
        files = list(folder.glob('*'))
        if files:
            samples = random.sample(files, min(3, len(files)))
            print(f"\n{name}:")
            for f in samples:
                with Image.open(f) as img:
                    print(f"{f.name}: {img.size}")
        else:
            print(f"\n{name}: No files found")

def resize_dataset(args):
    """Step 4: Resize images to target size with parallel processing"""
    print(f"\n--- Step 4: Resizing images to {args.target_size}x{args.target_size} ---")
    
    # Define paths
    source_dir = 'cyclegan_dataset_matched'
    target_dir = 'cyclegan_dataset_256'
    size = (args.target_size, args.target_size)

    # Create target directories
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Function to resize a single file
    def resize_file(args):
        file, target_dir, size = args
        try:
            # Load image
            with Image.open(file) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Use LANCZOS resampling for high quality
                resized = img.resize(size, Image.Resampling.LANCZOS)
                
                # Create subdirectory if needed
                subdir = file.parent.name
                output_dir = Path(target_dir) / subdir
                output_dir.mkdir(exist_ok=True)
                
                # Save to new location
                new_path = output_dir / file.name
                resized.save(new_path, quality=95)
                return new_path
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None
    
    # Process all three folders
    all_files = []
    for subdir in ['trainA', 'trainB', 'test_healthy']:
        # Create target subdirectory
        Path(target_dir, subdir).mkdir(exist_ok=True)
        
        # Get source files
        source_path = Path(source_dir, subdir)
        files = list(source_path.glob('*.png')) + list(source_path.glob('*.jpg'))
        all_files.extend([(f, target_dir, size) for f in files])
    
    # Process files in parallel
    print(f"\nResizing {len(all_files)} images in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(resize_file, all_files),
            total=len(all_files)
        ))
    
    # Count successful resizes
    successful = len([r for r in results if r])
    print(f"\nSuccessfully resized {successful} of {len(all_files)} images")

    # Verify the resized dataset
    print("\nVerifying resized dataset...")
    for subdir in ['trainA', 'trainB', 'test_healthy']:
        path = Path(target_dir, subdir)
        files = list(path.glob('*.png')) + list(path.glob('*.jpg'))

        print(f"\nChecking {subdir}...")
        sizes = set()
        for f in files[:10]:  # Just check a few files for verification
            with Image.open(f) as img:
                sizes.add(img.size)
        print(f"Unique sizes found: {sizes}")
        print(f"Total images: {len(files)}")

def split_dataset(args):
    """Step 5: Split dataset into train/val with parallel processing"""
    print("\n--- Step 5: Splitting dataset into train/val ---")
    
    # Configuration
    data_dir = 'cyclegan_dataset_256'
    target_dir = Path(str(data_dir) + '_split')
    test_split = args.test_split

    # Create directory structure
    for split in ['train', 'val']:
        for domain in ['A', 'B']:
            (target_dir / f'{split}{domain}').mkdir(parents=True, exist_ok=True)

    # Also create test_healthy directory
    (target_dir / 'test_healthy').mkdir(parents=True, exist_ok=True)

    # Function to copy a file to its destination
    def copy_file(args):
        src, dest = args
        try:
            shutil.copy(src, dest)
            return dest
        except Exception as e:
            print(f"Error copying {src} to {dest}: {e}")
            return None
    
    # Process both domains
    all_copy_tasks = []
    
    for domain in ['A', 'B']:
        files = list(Path(data_dir, f'train{domain}').glob('*.png')) + list(Path(data_dir, f'train{domain}').glob('*.jpg'))
        total_files = len(files)

        # Calculate split size
        test_size = int(total_files * test_split)

        # Randomly select files for each split
        test_files = random.sample(files, test_size)
        train_files = [f for f in files if f not in test_files]

        # Create copy tasks
        for f in train_files:
            all_copy_tasks.append((f, target_dir / f'train{domain}' / f.name))
        for f in test_files:
            all_copy_tasks.append((f, target_dir / f'val{domain}' / f.name))
    
    # Add test_healthy copy tasks
    test_healthy_files = list(Path(data_dir, 'test_healthy').glob('*.png')) + list(Path(data_dir, 'test_healthy').glob('*.jpg'))
    for f in test_healthy_files:
        all_copy_tasks.append((f, target_dir / 'test_healthy' / f.name))
    
    # Copy files in parallel
    print(f"\nCopying {len(all_copy_tasks)} files in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(copy_file, all_copy_tasks),
            total=len(all_copy_tasks)
        ))
    
    # Count successful copies
    successful = len([r for r in results if r])
    print(f"\nSuccessfully copied {successful} of {len(all_copy_tasks)} files")
    
    # Print dataset statistics
    print(f"\nDataset split created at {target_dir}")
    for dir_path in target_dir.glob('*'):
        files = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg'))
        print(f"{dir_path.name}: {len(files)} images")

def main():
    args = parse_args()
    
    steps = args.steps.lower().split(',')
    run_all = 'all' in steps
    
    stats = None
    
    if run_all or 'validate' in steps:
        stats = validate_create_combined(args)
        
    if run_all or 'augment' in steps:
        augment_create_matched(args, stats)
        
    if run_all or 'resize' in steps:
        resize_dataset(args)
        
    if run_all or 'split' in steps:
        split_dataset(args)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()