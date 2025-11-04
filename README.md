# ID2222 Data Mining - Document Similarity Project

This project implements document similarity analysis using:

- **Shingling**: k-shingle extraction and hashing
- **MinHashing**: MinHash signature generation
- **Locality-Sensitive Hashing (LSH)**: Efficient candidate pair detection
- **Jaccard Similarity**: Set and signature similarity computation

All implemented with Apache Spark for parallel processing.

## Prerequisites

- Python 3.7 or higher
- Apache Spark (PySpark)
- Java (required for Spark)

## Installation

1. Install PySpark:

```bash
pip install pyspark
```

Alternatively, if you're using a requirements file:

```bash
pip install -r requirements.txt
```

2. Ensure Java is installed (Spark requires Java):

```bash
java -version
```

If Java is not installed:

- **macOS**: `brew install openjdk`
- **Linux**: `sudo apt-get install default-jdk` (Ubuntu/Debian)
- **Windows**: Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or use [Adoptium](https://adoptium.net/)

## Running the Code

### Option 1: Simple Demo (main.py)

This runs a simple demo with hardcoded sample documents:

```bash
python main.py
```

### Option 2: Document Similarity Analysis (documentSimilarity.py)

This loads documents from the `Resources/Dataset` folder and computes Jaccard similarity:

```bash
python documentSimilarity.py
```

By default, it:

- Loads 10 random files from `Resources/Dataset`
- Creates shingles with k=9
- Computes Jaccard similarity between all pairs of documents
- Groups results into high similarity (≥0.8) and low similarity (<0.8)

### Option 3: Full Pipeline with MinHashing and LSH (fullPipeline.py)

This runs the complete pipeline including MinHashing and LSH:

```bash
python fullPipeline.py
```

This includes:

- Shingling
- MinHash signature computation
- Signature similarity comparison
- LSH for efficient candidate pair detection
- Verification of LSH candidates

### Option 4: Test All Classes (testClasses.py)

Test all individual classes to verify they work correctly:

```bash
python testClasses.py
```

### Customizing Parameters

You can modify the parameters in `documentSimilarity.py`:

```python
if __name__ == '__main__':
    base_path = "Resources/Dataset"  # Change dataset path
    app = DocumentSimilarity(k=9)     # Change shingle size (k)
    app.run(base_path, num_files=10)  # Change number of files to process
```

## Project Structure

### Core Classes (Required by Assignment)

- `shingling.py` - **Shingling class**: Creates k-shingles from documents and hashes them
- `compareSets.py` - **CompareSets class**: Computes Jaccard similarity between two sets of shingles
- `minHashing.py` - **MinHashing class**: Builds minHash signatures from shingle sets
- `compareSignatures.py` - **CompareSignatures class**: Estimates similarity of two minHash signatures
- `lsh.py` - **LSH class** (bonus): Implements Locality-Sensitive Hashing for efficient candidate pair detection

### Applications

- `main.py` - Simple demo application with hardcoded documents
- `documentSimilarity.py` - Basic document similarity analysis using shingling and Jaccard similarity
- `fullPipeline.py` - Complete pipeline using all stages (Shingling → MinHashing → LSH)
- `testClasses.py` - Unit tests for all classes

### Data

- `Resources/Dataset/` - Contains the document files to analyze

## How It Works

### Stage 1: Shingling

- Converts each document into a set of k-shingles (k-character subsequences)
- Normalizes text (lowercase, removes punctuation)
- Hashes shingles to integers using MD5

### Stage 2: MinHashing

- Generates n random hash functions
- For each hash function, finds the minimum hash value across all shingles
- Creates a signature vector of length n

### Stage 3: LSH (Locality-Sensitive Hashing)

- Divides signatures into bands
- Hashes each band to find candidate pairs
- Only documents with matching bands are considered candidates
- Significantly reduces the number of comparisons needed

### Stage 4: Similarity Computation

- **Jaccard Similarity**: Direct comparison of shingle sets
- **Signature Similarity**: Fraction of matching components in minHash signatures
- Both methods provide similarity scores between 0.0 and 1.0

### Parallel Processing

- Uses Spark RDDs to process documents in parallel
- Efficient for large document collections

## Example Usage

```python
from shingling import Shingling
from compareSets import CompareSets
from minHashing import MinHashing
from compareSignatures import CompareSignatures
from lsh import LSH

# Create shingles
shingler = Shingling(k=9)
shingles1 = shingler.create_shingles("Document 1 text...")
shingles2 = shingler.create_shingles("Document 2 text...")

# Compare sets
jaccard_sim = CompareSets.jaccard_similarity(shingles1, shingles2)

# Create minHash signatures
minhasher = MinHashing(n=100)
sig1 = minhasher.compute_signature(shingles1)
sig2 = minhasher.compute_signature(shingles2)

# Compare signatures
sig_sim = CompareSignatures.similarity(sig1, sig2)

# Use LSH for efficient candidate detection
lsh = LSH(threshold=0.8)
signatures = [sig1, sig2, ...]  # List of all signatures
candidate_pairs = lsh.find_candidate_pairs(signatures)
```
