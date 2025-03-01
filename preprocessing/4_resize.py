from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

# Define paths
source_dir = '/cyclegan_dataset_matched'
target_dir = '/cyclegan_dataset_256'
size = (256, 256)

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