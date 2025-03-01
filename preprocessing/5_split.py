import random
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Configuration
data_dir = 'cyclegan_dataset_256'
target_dir = Path(str(data_dir) + '_split')
test_split = 0.1

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