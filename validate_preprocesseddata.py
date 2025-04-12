#!/usr/bin/env python3

import os
import subprocess
import sys
import shlex

# --- Configuration ---
# Define the expected directories and their human-readable sizes based on 'du -hl' output.
# Paths are relative to the directory where the script is run (expected to be papGAN/).
EXPECTED_DATA = [
    # isbi2025 dataset parts
    {"path": "isbi2025-ps3c-train-dataset/isbi2025-ps3c-train-dataset/bothcells", "size": "694M"},
    {"path": "isbi2025-ps3c-train-dataset/isbi2025-ps3c-train-dataset/healthy", "size": "5.4G"},
    {"path": "isbi2025-ps3c-train-dataset/isbi2025-ps3c-train-dataset/rubbish", "size": "8.8G"},
    {"path": "isbi2025-ps3c-train-dataset/isbi2025-ps3c-train-dataset/unhealthy", "size": "385M"},
    {"path": "isbi2025-ps3c-train-dataset/isbi2025-ps3c-train-dataset/both_and_unhealthy", "size": "1.1G"},
    # Parent isbi2025 dataset dir (Note: Size might fluctuate slightly)
    {"path": "isbi2025-ps3c-train-dataset/isbi2025-ps3c-train-dataset", "size": "17G"},

    # cyclegan dataset parts
    # For "0" size, we primarily check existence and that it's a directory.
    # 'du -sh' on an empty dir might report 0 or a small size like 4.0K.
    # We'll check existence mainly for these.
    {"path": "cyclegan_dataset_matched/trainA", "size": "0"}, # Expect to exist, maybe empty
    {"path": "cyclegan_dataset_matched/trainB", "size": "2.1G"},
    {"path": "cyclegan_dataset_matched/test_healthy", "size": "0"}, # Expect to exist, maybe empty
    # Parent cyclegan dataset dir
    {"path": "cyclegan_dataset_matched", "size": "2.1G"},
]

# --- Helper Function ---
def get_dir_size_human(dir_path):
    """
    Gets the directory size using 'du -sh' command and returns the human-readable size string.
    Returns None if the directory doesn't exist or an error occurs.
    """
    if not os.path.isdir(dir_path):
        return None # Indicate directory not found or not a directory

    try:
        # Use shlex.quote to handle potential special characters in paths, though unlikely here
        command = f"du -sh {shlex.quote(dir_path)}"
        result = subprocess.run(
            command,
            shell=True, # Using shell=True because of the simple command structure
            capture_output=True,
            text=True,
            check=True # Raise CalledProcessError on non-zero exit code
        )
        # Output is typically like: "5.4G\t./path/to/dir" or just "5.4G\t."
        size_str = result.stdout.split()[0]
        return size_str
    except FileNotFoundError:
        print(f"Error: 'du' command not found. Make sure it's installed and in your PATH.", file=sys.stderr)
        return "ERROR_DU_NOT_FOUND"
    except subprocess.CalledProcessError as e:
        print(f"Error running 'du' on {dir_path}: {e.stderr}", file=sys.stderr)
        return "ERROR_DU_FAILED"
    except Exception as e:
        print(f"An unexpected error occurred while getting size for {dir_path}: {e}", file=sys.stderr)
        return "ERROR_UNEXPECTED"

# --- Main Validation Logic ---
def main():
    print("Starting preprocessed data validation...")
    print(f"Running checks relative to: {os.getcwd()}")
    print("-" * 30)

    all_ok = True
    errors = []
    base_dir = "." # Assume script is run from the parent (papGAN) directory

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

        # 2. Handle expected "0" size (mainly check existence, which we did)
        if expected_size == "0":
            # We've already confirmed it exists and is a directory.
            # You could add a check here to ensure it's truly empty or very small if needed
            # e.g., check len(os.listdir(path_to_check)) == 0
            print(f"OK (Directory exists as expected)")
            continue

        # 3. Check size for non-zero expectations using 'du -sh'
        actual_size = get_dir_size_human(path_to_check)

        if actual_size is None:
             # This case should theoretically not happen if os.path.isdir passed,
             # but handle defensively. Error would likely be logged by get_dir_size_human.
            error_msg = f"FAILED: Could not determine size for existing directory '{path_to_check}' (isdir check passed but get_dir_size_human failed)."
            print(error_msg)
            errors.append(error_msg)
            all_ok = False
        elif "ERROR" in actual_size: # Check for specific error codes from helper
             error_msg = f"FAILED: Error occurred while getting size for '{path_to_check}' (check logs above)."
             print(error_msg)
             errors.append(error_msg) # Error already printed
             all_ok = False
        elif actual_size == expected_size:
            print(f"OK (Size: {actual_size})")
        else:
            # Simple string comparison failed.
            error_msg = f"FAILED: Size mismatch for '{path_to_check}'. Expected: {expected_size}, Actual: {actual_size}"
            print(error_msg)
            # Add a note about potential acceptable variations
            print("       Note: Slight variations in reported size might be acceptable depending on filesystem.")
            errors.append(error_msg)
            all_ok = False

    # --- Summary ---
    print("-" * 30)
    print("\nValidation Summary:")
    if all_ok:
        print("✅ All checks passed. Preprocessed data structure and sizes appear OK.")
        sys.exit(0)
    else:
        print("❌ Validation FAILED. Issues found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()