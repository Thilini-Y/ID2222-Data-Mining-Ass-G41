# Generate a larger test dataset by copying/varying existing documents
# This allows testing scalability with 1000+ files
import os
import shutil
import random


def generate_test_dataset(source_path, target_path, num_files=1000):
    """
    Generate a test dataset with specified number of files.
    Creates copies and variations of existing documents.
    """
    if not os.path.exists(source_path):
        print(f"Error: Source folder not found: {source_path}")
        return False

    # Get all source files
    source_files = [
        f
        for f in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, f)) and not f.startswith(".")
    ]

    if not source_files:
        print(f"Error: No files found in {source_path}")
        return False

    # Create target directory
    os.makedirs(target_path, exist_ok=True)

    print(f"Generating {num_files} files from {len(source_files)} source files...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print()

    # Generate files
    for i in range(num_files):
        # Randomly select a source file
        source_file = random.choice(source_files)
        source_path_full = os.path.join(source_path, source_file)

        # Create target filename
        base_name = os.path.splitext(source_file)[0]
        target_file = f"{base_name}_test_{i:04d}"
        target_path_full = os.path.join(target_path, target_file)

        # Copy the file
        shutil.copy2(source_path_full, target_path_full)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_files} files...")

    print(f"\nDone! Generated {num_files} files in {target_path}")
    return True


if __name__ == "__main__":
    # Generate 1000 test files
    source = "Resources/Dataset"  # Your original dataset
    target = "Resources/Dataset_Large"  # Large test dataset

    print("=" * 70)
    print("GENERATING LARGE TEST DATASET")
    print("=" * 70)
    print()

    success = generate_test_dataset(source, target, num_files=1000)

    if success:
        print()
        print("=" * 70)
        print("Next steps:")
        print(f"1. Update scalabilityTest.py to use: base_path = '{target}'")
        print("2. Run: python3 scalabilityTest.py")
        print("=" * 70)
