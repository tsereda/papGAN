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
    """Step 1: Validate dataset and create combined unhealthy folder"""
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

    # Copy images from 'both' and 'unhealthy' folders
    sources = ['bothcells', 'unhealthy']
    copied_count = 0

    for source in sources:
        source_path = f'{args.source_dir}/{source}'
        print(f"\nChecking source path: {source_path}")
        print(f"Is path in stats? {source_path in stats}")
        if source_path in stats:
            print(f"Number of files found: {len(stats[source_path]['files'])}")
            for file in stats[source_path]['files']:
                dest_file = os.path.join(combined_path, f"{source}_{file.name}")
                shutil.copy2(file, dest_file)
                copied_count += 1

    print(f"\nCreated combined folder: {combined_path}")
    print(f"Copied {copied_count} images")
    
    return stats

def augment_create_matched(args, stats=None):
    """Step 3: Augment data and create matched dataset"""
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

    # Get target size from unhealthy images
    print('Analyzing unhealthy images...')
    unhealthy_sizes = np.array([Image.open(f).size for f in tqdm(unhealthy_files)])
    target_size = unhealthy_sizes.mean(axis=0)

    # Copy and augment unhealthy files
    print('\nCopying and augmenting unhealthy files...')
    file_counter = 0
    for src in tqdm(unhealthy_files):
        # Copy original
        img = Image.open(src)
        shutil.copy(src, trainB / f'unhealthy_{file_counter:04d}{src.suffix}')
        file_counter += 1

        # Apply one random transform
        transform = random.choice(transform_list)
        aug_img = transform(img)
        aug_img.save(trainB / f'unhealthy_aug_{file_counter:04d}{src.suffix}')
        file_counter += 1

    # Get actual count of trainB after augmentation
    trainB_count = len(list(trainB.glob('*')))
    print(f'\nTotal unhealthy images after augmentation: {trainB_count}')

    # Select matching healthy images
    print('\nSelecting size-matched healthy images...')
    healthy_matches = []
    for f in tqdm(healthy_files):
        size = np.array(Image.open(f).size)
        healthy_matches.append((np.linalg.norm(size - target_size), f))

    # Sort by size similarity and split into train and test
    sorted_healthy = sorted(healthy_matches, key=lambda x: x[0])
    selected_healthy = [f for _, f in sorted_healthy[:trainB_count]]
    test_healthy_files = [f for _, f in sorted_healthy[trainB_count:]]

    # Copy train healthy files
    print('\nCopying training healthy files...')
    for i, src in enumerate(tqdm(selected_healthy)):
        shutil.copy(src, trainA / f'healthy_{i:04d}{src.suffix}')

    # Copy test healthy files
    print('\nCopying test healthy files...')
    for i, src in enumerate(tqdm(test_healthy_files)):
        shutil.copy(src, test_healthy / f'healthy_{i:04d}{src.suffix}')

    print(f'\nDataset created at {output_dir}:')
    print(f'trainA (healthy): {len(list(trainA.glob("*")))} images')
    print(f'trainB (unhealthy+bothcells): {len(list(trainB.glob("*")))} images')
    print(f'test_healthy: {len(list(test_healthy.glob("*")))} images')

    # Verify sample sizes
    print("\nSample image sizes:")
    for folder in [trainA, trainB, test_healthy]:
        files = list(folder.glob('*'))
        samples = random.sample(files, min(3, len(files)))
        print(f"\n{folder.name}:")
        for f in samples:
            with Image.open(f) as img:
                print(f"{f.name}: {img.size}")

def resize_dataset(args):
    """Step 4: Resize images to target size"""
    print(f"\n--- Step 4: Resizing images to {args.target_size}x{args.target_size} ---")
    
    # Define paths
    source_dir = 'cyclegan_dataset_matched'
    target_dir = 'cyclegan_dataset_256'
    size = (args.target_size, args.target_size)

    # Create target directories
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Process all three folders
    for subdir in ['trainA', 'trainB', 'test_healthy']:
        # Create target subdirectory
        Path(target_dir, subdir).mkdir(exist_ok=True)

        # Get source files
        source_path = Path(source_dir, subdir)
        files = list(source_path.glob('*.png')) + list(source_path.glob('*.jpg'))

        print(f"\nProcessing {subdir}...")
        for file in tqdm(files):
            # Load and resize image
            with Image.open(file) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Use LANCZOS resampling for high quality
                resized = img.resize(size, Image.Resampling.LANCZOS)

                # Save to new location
                new_path = Path(target_dir, subdir, file.name)
                resized.save(new_path, quality=95)

    # Verify the resized dataset
    print("\nVerifying resized dataset...")
    for subdir in ['trainA', 'trainB', 'test_healthy']:
        path = Path(target_dir, subdir)
        files = list(path.glob('*.png')) + list(path.glob('*.jpg'))

        print(f"\nChecking {subdir}...")
        sizes = set()
        for f in files:
            with Image.open(f) as img:
                sizes.add(img.size)
        print(f"Unique sizes found: {sizes}")
        print(f"Total images: {len(files)}")

def split_dataset(args):
    """Step 5: Split dataset into train/val"""
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

    # Split and move files
    for domain in ['A', 'B']:
        files = list(Path(data_dir, f'train{domain}').glob('*.png'))
        total_files = len(files)

        # Calculate split size
        test_size = int(total_files * test_split)

        # Randomly select files for each split
        test_files = random.sample(files, test_size)
        train_files = [f for f in files if f not in test_files]

        # Move files
        for split, file_list in [
            ('train', train_files),
            ('val', test_files)
        ]:
            for f in file_list:
                shutil.copy(f, target_dir / f'{split}{domain}' / f.name)

    # Copy test_healthy directory
    test_healthy_files = list(Path(data_dir, 'test_healthy').glob('*.png'))
    for f in test_healthy_files:
        shutil.copy(f, target_dir / 'test_healthy' / f.name)
    
    print(f"\nDataset split created at {target_dir}")
    for dir_path in target_dir.glob('*'):
        files = list(dir_path.glob('*.png'))
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