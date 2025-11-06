import os
import shutil
from pathlib import Path

# Define paths
csv_data_dir = "Resources/CSV_Data"
talk_religion_dir = "Resources/talk.religion.misc"
output_dir = "Resources/Total_data"
target_count = 1000

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get first 500 files from CSV_Data
csv_files_all = sorted(Path(csv_data_dir).glob("*.txt"))
csv_files = csv_files_all[:500]  # First 500 as specified
print(
    f"Found {len(csv_files_all)} files in {csv_data_dir}, will use first {len(csv_files)}"
)

# Get first 500 files from talk.religion.misc
talk_religion_files_all = sorted(Path(talk_religion_dir).iterdir())
talk_religion_files_all = [f for f in talk_religion_files_all if f.is_file()]
talk_religion_files = talk_religion_files_all[:500]  # First 500 as specified
print(
    f"Found {len(talk_religion_files_all)} files in {talk_religion_dir}, will use first {len(talk_religion_files)}"
)

copied_files = []
file_counter = 1

# Copy first 500 CSV files
for csv_file in csv_files:
    dest_file = os.path.join(output_dir, f"doc_{file_counter:04d}.txt")
    shutil.copy2(csv_file, dest_file)
    copied_files.append(dest_file)
    file_counter += 1

print(f"Copied {len(csv_files)} files from {csv_data_dir}")

# Copy first 500 talk.religion files
for talk_file in talk_religion_files:
    dest_file = os.path.join(output_dir, f"doc_{file_counter:04d}.txt")
    shutil.copy2(talk_file, dest_file)
    copied_files.append(dest_file)
    file_counter += 1

print(f"Copied {len(talk_religion_files)} files from {talk_religion_dir}")
print(f"Total files copied: {len(copied_files)}")

# If we have more than target_count, remove excess files
if len(copied_files) > target_count:
    print(
        f"Found {len(copied_files)} files. Removing excess files to reach {target_count}..."
    )
    for excess_file in copied_files[target_count:]:
        os.remove(excess_file)
    copied_files = copied_files[:target_count]
    # Renumber remaining files to be sequential
    for i, file_path in enumerate(copied_files, start=1):
        new_name = os.path.join(output_dir, f"doc_{i:04d}.txt")
        if file_path != new_name:
            os.rename(file_path, new_name)
            copied_files[i - 1] = new_name

# If we still have less than target_count, duplicate files cyclically until we reach target_count
if len(copied_files) < target_count:
    current_count = len(copied_files)
    print(
        f"Only {current_count} files copied. Duplicating files to reach {target_count}..."
    )
    source_files = copied_files.copy()  # Files to cycle through
    source_index = 0

    while current_count < target_count:
        source_file = source_files[source_index % len(source_files)]
        dest_file = os.path.join(output_dir, f"doc_{current_count + 1:04d}.txt")
        shutil.copy2(source_file, dest_file)
        copied_files.append(dest_file)
        current_count += 1
        source_index += 1

print(f"✅ Created {len(copied_files)} files in ./{output_dir}/")
