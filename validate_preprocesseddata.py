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
    print("-" * 30)

    all_ok = True
    errors = []

    for item in EXPECTED_DATA:
        path_to_check = os.path.join(base_dir, item["path"])
        expected_size = item["size"]
        print(f"Checking: {path_to_check} ...")

        # 1. Check if path exists and is a directory
        if not os.path.exists(path_to_check):
            error_msg = f"FAILED: Directory '{path_to_check}' does not exist."
            print(error_msg)
            errors.append(error_msg)
            all_ok = False
            continue # Skip size check if directory doesn't exist

        if not os.path.isdir(path_to_check):
            error_msg = f"FAILED: Path '{path_to_check}' exists but is not a directory."
            print(error_msg)
            errors.append(error_msg)
            all_ok = False
            continue # Skip size check if not a directory

        # For directories that should exist but size doesn't matter
        if expected_size == "0":
            # Check if it contains files
            if len(os.listdir(path_to_check)) == 0:
                warning_msg = f"WARNING: Directory '{path_to_check}' exists but is empty."
                print(warning_msg)
                if item["path"] != "cyclegan_dataset_256_split":  # Parent dir can be empty
                    errors.append(warning_msg)
                    all_ok = False
            else:
                print(f"OK (Directory exists and contains files)")
            continue

        # We can add size checks if needed later

    # --- Summary ---
    print("-" * 30)
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