#print stats
import os
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

# Define paths
paths = [
    'isbi2025-ps3c-train-dataset/healthy',
    'isbi2025-ps3c-train-dataset/unhealthy',
    'isbi2025-ps3c-train-dataset/bothcells',
    'isbi2025-ps3c-train-dataset/rubbish'
]

# Initialize stats dictionary
stats = {}

# Analyze each path
for path in paths:
    print(f"\nChecking path: {path}")  # Debug print
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        continue

    files = list(Path(path).glob('*.jpg')) + list(Path(path).glob('*.png'))
    print(f"Found {len(files)} files in {path}")  # Debug print
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
combined_path = 'isbi2025-ps3c-train-dataset/both_and_unhealthy'
os.makedirs(combined_path, exist_ok=True)

# Copy images from 'both' and 'unhealthy' folders
sources = ['bothcells', 'unhealthy']
copied_count = 0

for source in sources:
    source_path = f'isbi2025-ps3c-train-dataset/{source}'
    print(f"\nChecking source path: {source_path}")  # Debug print
    print(f"Is path in stats? {source_path in stats}")  # Debug print
    if source_path in stats:
        print(f"Number of files found: {len(stats[source_path]['files'])}")  # Debug print
        for file in stats[source_path]['files']:
            dest_file = os.path.join(combined_path, f"{source}_{file.name}")
            shutil.copy2(file, dest_file)
            copied_count += 1

print(f"\nCreated combined folder: {combined_path}")
print(f"Copied {copied_count} images")
