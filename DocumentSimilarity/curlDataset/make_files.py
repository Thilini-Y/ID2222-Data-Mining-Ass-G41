import csv
import os

# Define paths
csv_file = "sts-test.csv"
output_dir = "Resources/CSV_Data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read CSV and extract 6th column (index 5)
with open(csv_file, "r", encoding="utf-8") as f:
    # Use tab delimiter since the CSV appears to be tab-separated
    reader = csv.reader(f, delimiter="\t")

    for row_num, row in enumerate(reader, start=1):
        if len(row) > 5:  # Make sure row has at least 6 columns
            # Get 6th column (index 5)
            text_content = row[5]

            # Create filename based on row number
            filename = f"doc_{row_num:04d}.txt"
            filepath = os.path.join(output_dir, filename)

            # Write content to file
            with open(filepath, "w", encoding="utf-8") as out_file:
                out_file.write(text_content)

            if row_num % 100 == 0:
                print(f"Processed {row_num} files...")

print(f"✅ Saved {row_num} files to ./{output_dir}/")
