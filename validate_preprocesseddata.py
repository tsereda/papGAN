#!/usr/bin/env python3

import os
import subprocess
import sys
import shlex
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Validate the preprocessed data directory structure.')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory containing the dataset directories')
    return parser.parse_args()

# --- Helper function to count files recursively ---
def count_files(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

# --- Main Validation Logic ---
def main():
    args = parse_args()
    base_dir = args.base_dir
    
    # Update with your actual directory structure
    EXPECTED_DATA = [
        # cyclegan dataset parts
        {"path": "cyclegan_dataset_256_split/trainA", "size": "0"},
        {"path": "cyclegan_dataset_256_split/trainB", "size": "0"},
        {"path": "cyclegan_dataset_256_split/test_healthy", "size": "0"},
        {"path": "cyclegan_dataset_256_split/valA", "size": "0"},
        {"path": "cyclegan_dataset_256_split/valB", "size": "0"},
        # Parent cyclegan dataset dir
        {"path": "cyclegan_dataset_256_split", "size": "0"},
    ]
    
    print("Starting preprocessed data validation...")
    print(f"Running checks relative to: {os.path.abspath(base_dir)}")
    print("-" * 50)
    print(f"{'Directory':<40} {'Status':<15} {'Files':<10}")
    print("-" * 50)

    all_ok = True
    errors = []
    total_files_by_dir = {}

    for item in EXPECTED_DATA:
        path_to_check = os.path.join(base_dir, item["path"])
        expected_size = item["size"]
        
        # Skip the parent directory for now, we'll process it last
        if item["path"] == "cyclegan_dataset_256_split":
            continue
            
        # 1. Check if path exists and is a directory
        if not os.path.exists(path_to_check):
            error_msg = f"FAILED: Directory '{path_to_check}' does not exist."
            print(f"{path_to_check:<40} {'FAILED':<15} {'N/A':<10}")
            errors.append(error_msg)
            all_ok = False
            continue # Skip further checks if directory doesn't exist

        if not os.path.isdir(path_to_check):
            error_msg = f"FAILED: Path '{path_to_check}' exists but is not a directory."
            print(f"{path_to_check:<40} {'FAILED':<15} {'N/A':<10}")
            errors.append(error_msg)
            all_ok = False
            continue # Skip further checks if not a directory

        # Count files
        file_count = count_files(path_to_check)
        total_files_by_dir[item["path"]] = file_count
        
        if file_count == 0:
            warning_msg = f"WARNING: Directory '{path_to_check}' exists but is empty."
            print(f"{path_to_check:<40} {'WARNING':<15} {file_count:<10}")
            errors.append(warning_msg)
            all_ok = False
        else:
            print(f"{path_to_check:<40} {'OK':<15} {file_count:<10}")

    # Now check the parent directory
    parent_dir = os.path.join(base_dir, "cyclegan_dataset_256_split")
    if os.path.isdir(parent_dir):
        total_files = count_files(parent_dir)
        # The sum should match the individual directories' total
        sum_individual = sum(total_files_by_dir.values())
        print(f"{parent_dir:<40} {'OK':<15} {total_files:<10}")
        print("-" * 50)
        print(f"Total files in parent directory: {total_files}")
        print(f"Sum of files in subdirectories: {sum_individual}")
        
        # There might be a difference if there are files directly in the parent directory
        if total_files != sum_individual:
            print(f"Note: Difference of {total_files - sum_individual} files directly in parent directory")
    
    # --- Summary ---
    print("-" * 50)
    print("\nValidation Summary:")
    if all_ok:
        print("✅ All checks passed. Preprocessed data structure appears OK.")
        sys.exit(0)
    else:
        print("❌ Validation FAILED. Issues found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()