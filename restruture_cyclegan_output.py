import os
import shutil
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Sort and rename image files')
parser.add_argument('input_folder', type=str, help='Path to the folder containing images')
args = parser.parse_args()

# Get absolute path of input folder
input_folder = os.path.abspath(args.input_folder)
parent_dir = os.path.dirname(input_folder)

# Create output directories at the same level as input folder
input_dir = os.path.join(parent_dir, "input")
generated_dir = os.path.join(parent_dir, "generated")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Process all PNG files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        source_path = os.path.join(input_folder, filename)
        
        # Check if it's a fake or real image
        if "_fake.png" in filename:
            # Extract the number part
            number_part = filename.split("_")[1]
            # Create new filename for fake images
            new_filename = f"fake_unhealthy_{number_part}.png"
            # Move to generated directory
            shutil.copy2(source_path, os.path.join(generated_dir, new_filename))
        elif "_real.png" in filename:
            # Extract the number part
            number_part = filename.split("_")[1]
            # Create new filename for real images
            new_filename = f"real_healthy_{number_part}.png"
            # Move to input directory
            shutil.copy2(source_path, os.path.join(input_dir, new_filename))

print("Images have been sorted and renamed successfully!")
print(f"Real images are in: {input_dir}")
print(f"Fake images are in: {generated_dir}")