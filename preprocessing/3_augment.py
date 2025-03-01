import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm
import random
from torchvision import transforms

# Setup paths
healthy_dir = 'isbi2025-ps3c-train-dataset/healthy'
combined_dir = 'isbi2025-ps3c-train-dataset/both_and_unhealthy'
output_dir = Path('cyclegan_dataset_matched')

trainA = output_dir / 'trainA'
trainB = output_dir / 'trainB'
test_healthy = output_dir / 'test_healthy'  # New directory for test healthy images
trainA.mkdir(parents=True, exist_ok=True)
trainB.mkdir(parents=True, exist_ok=True)
test_healthy.mkdir(parents=True, exist_ok=True)  # Create test directory

# Define augmentation transforms
transform_list = [
    transforms.RandomHorizontalFlip(p=1.0),  # Horizontal flip
    transforms.RandomVerticalFlip(p=1.0),     # Vertical flip
    transforms.RandomRotation(90),            # 90-degree rotation
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
test_healthy_files = [f for _, f in sorted_healthy[trainB_count:]]  # Remaining images go to test

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