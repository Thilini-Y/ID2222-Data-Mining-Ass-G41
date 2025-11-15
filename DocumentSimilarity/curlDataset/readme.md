# Dataset Creation Guide

This directory contains scripts for creating and organizing datasets from various sources.

## Overview

The dataset creation process involves:

1. Converting CSV data to individual text files
2. Combining files from multiple sources into a unified dataset
3. Creating a final dataset of 1000 files

## Files

- `make_files.py` - Converts CSV data to individual text files
- `total_dataset.py` - Combines files from multiple sources into a final dataset

## Step 1: Download and Prepare CSV Data

Download the STS test CSV file from:
https://github.com/thisisclement/STS-Benchmark-SentEval/blob/master/data/sts-test.csv

Save it as `sts-test.csv` in the main directory (same level as the `curlDataset` folder).

## Step 2: Convert CSV to Text Files

Run `make_files.py` to extract the 6th column from the CSV and save each row as a separate text file:

```bash
python make_files.py
```

This will:

- Read `sts-test.csv` from the main directory
- Extract the 6th column (index 5) from each row
- Save each value as a separate `.txt` file in `Resources/CSV_Data/`
- Files are named `doc_0001.txt`, `doc_0002.txt`, etc.

**Output:** `Resources/CSV_Data/` (contains ~1119 text files)

## Step 3: Prepare talk.religion.misc Dataset

Download the 20 Newsgroups dataset from:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

Extract and move the `talk.religion.misc` folder to:

```
Resources/talk.religion.misc/
```

This folder should contain approximately 1000 newsgroup files.

## Step 4: Create Final Combined Dataset

Run `total_dataset.py` to combine files from both sources:

```bash
python total_dataset.py
```

This will:

- Copy the first 500 files from `Resources/CSV_Data/`
- Copy the first 500 files from `Resources/talk.religion.misc/`
- Save all files in `Resources/Total_data/`
- Number files sequentially from `doc_0001.txt` to `doc_1000.txt`

**Output:** `Resources/Total_data/` (contains exactly 1000 text files)

## Final Dataset Structure

```
Resources/Total_data/
├── doc_0001.txt   (from CSV_Data)
├── doc_0002.txt   (from CSV_Data)
├── ...
├── doc_0500.txt   (from CSV_Data)
├── doc_0501.txt   (from talk.religion.misc)
├── doc_0502.txt   (from talk.religion.misc)
├── ...
└── doc_1000.txt   (from talk.religion.misc)
```

## Notes

- All scripts should be run from the main project directory
- The scripts will create output directories automatically if they don't exist
- Files are encoded in UTF-8
- The final dataset contains exactly 1000 files (500 from each source)
