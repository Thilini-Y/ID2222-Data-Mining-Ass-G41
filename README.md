## Document Similarity Pipeline

This project implements a complete document-similarity workflow that combines shingling, MinHash signatures and Locality-Sensitive Hashing (LSH) on top of Apache Spark. The main entry point is `documentSimilarity.py`, which loads a subset of the documents under `Resources/Dataset`, computes exact Jaccard scores, estimates similarities with MinHash, and finally uses LSH to surface highly similar candidate pairs.

## Prerequisites

- **Python** 3.9 or newer
- **Java Runtime Environment** (required by PySpark). On macOS you can install it with `brew install openjdk`.
- **pip** for installing Python packages

## Setup

1. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project dependencies:
   ```bash
   pip install pyspark numpy
   ```
3. Ensure the dataset is available under `Resources/Dataset/`. The script expects the folder structure already provided in this repository; if you use your own data, place the files there or update the `base_path` in `documentSimilarity.py`.

## Preparing the dataset (from raw sources)

Run the following commands from the project root if you want to recreate the dataset from the raw sources in `curlDataset/`:

1. Download the STS test CSV file (`sts-test.csv`) from the [STS Benchmark repository](https://github.com/thisisclement/STS-Benchmark-SentEval/blob/master/data/sts-test.csv) and place it in the project root (same level as `curlDataset/`).
2. Convert the CSV rows into individual text files:
   ```bash
   python curlDataset/make_files.py
   ```
   - Produces ~1119 files in `Resources/CSV_Data/`.
3. Download the Twenty Newsgroups dataset, extract the `talk.religion.misc` subset, and place it under `Resources/talk.religion.misc/`.
4. Combine both sources into a 1000-document corpus:
   ```bash
   python curlDataset/total_dataset.py
   ```
   - Copies 500 files from `Resources/CSV_Data/` and 500 from `Resources/talk.religion.misc/` into `Resources/Total_data/`.

`documentSimilarity.py` loads data from `Resources/Dataset/` by default. After running the steps above, either copy the generated files from `Resources/Total_data/` into `Resources/Dataset/`, or update the `base_path` argument in the script to point to your preferred folder.

## Running `documentSimilarity.py`

The script selects a random subset of the documents, generates shingles, builds MinHash signatures, estimates similarities and runs LSH. By default it uses:

- `k = 9` (length of shingles)
- `num_perm = 1000` MinHash permutations
- `base_path = "Resources/Dataset"`
- `num_files = 10` randomly sampled documents

Run it with:

```bash
python documentSimilarity.py
```

## Customizing the run

Open `documentSimilarity.py` and adjust the block under `if __name__ == "__main__":` to change:

- `base_path`: point to another folder with text documents.
- `k` or `num_perm`: control shingle size and signature length.
- `num_files`: number of files sampled from the dataset.

Make sure that `bands * rows_per_band` inside the script matches `num_perm` when configuring LSH.

## Output

The script prints:

- Number of shingles per sampled document (first few hashes for inspection).
- Exact Jaccard similarity for every document pair, grouped into high/low similarity.
- MinHash-based similarity estimates for every pair.
- LSH candidate pairs that exceed the similarity threshold implied by the banding configuration.

The Spark session is created in local mode (`local[*]`) so the pipeline runs entirely on your machine.
