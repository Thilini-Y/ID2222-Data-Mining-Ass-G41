# ID2222 Data Mining - Document Similarity Project

This project implements document similarity analysis using shingling, minhashing, and locality-sensitive hashing (LSH) techniques. The implementation finds textually similar documents based on Jaccard similarity using Apache Spark for parallel processing.

## 📋 Assignment Requirements

This project implements the following stages:

1. ✅ **Shingling** - Constructs k-shingles from documents and computes hash values
2. ✅ **CompareSets** - Computes Jaccard similarity of two sets of hashed shingles
3. ✅ **MinHashing** - Builds minHash signatures from shingle sets
4. ✅ **CompareSignatures** - Estimates similarity of two minHash signatures
5. ✅ **LSH (Bonus)** - Implements Locality-Sensitive Hashing for efficient candidate pair detection

The project tests and evaluates scalability by measuring execution time versus dataset size (5-10+ documents) with a similarity threshold of 0.8.

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Apache Spark (PySpark)
- Java (required for Spark)

### Installation

1. **Install PySpark:**

```bash
pip3 install pyspark
```

2. **Verify Java is installed:**

```bash
java -version
```

If Java is not installed:

- **macOS**: `brew install openjdk`
- **Linux**: `sudo apt-get install default-jdk`
- **Windows**: Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [Adoptium](https://adoptium.net/)

## 📁 Project Structure

```
ID2222-Data-Minig/
├── Core Classes (Required)
│   ├── shingling.py              # Shingling class - creates k-shingles
│   ├── compareSets.py             # CompareSets class - Jaccard similarity
│   ├── minHashing.py              # MinHashing class - minHash signatures
│   ├── compareSignatures.py       # CompareSignatures class - signature similarity
│   └── lsh.py                     # LSH class (bonus) - efficient candidate detection
│
├── Applications
│   ├── main.py                    # Simple demo with hardcoded documents
│   ├── documentSimilarity.py      # Basic similarity analysis
│   ├── fullPipeline.py            # Complete pipeline (all stages)
│   ├── testClasses.py             # Unit tests for all classes
│   └── scalabilityTest.py         # Scalability testing (execution time vs dataset size)
│
├── Utilities
│   └── generateTestDataset.py    # Generate large test datasets
│
├── Documentation
│   ├── README.md                  # This file
│   ├── TEST_COMMANDS.md           # Detailed test commands
│   └── DATASET_RECOMMENDATIONS.md # Dataset suggestions
│
└── Resources/
    └── Dataset/                   # Document files for testing
```

## 🎯 Running the Code

### 1. Test All Classes (Recommended First Step)

```bash
python3 testClasses.py
```

Tests all individual classes with sample data to verify they work correctly.

### 2. Simple Demo

```bash
python3 main.py
```

Simple demo with 3 hardcoded sample documents using Spark.

### 3. Document Similarity Analysis (Basic)

```bash
python3 documentSimilarity.py
```

- Loads documents from `Resources/Dataset`
- Uses Shingling and Jaccard similarity
- Processes 10 random files by default
- Shows high/low similarity pairs (threshold = 0.8)

**Output:**

- Lists documents processed
- Shows number of shingles per document
- Groups pairs by high similarity (≥0.8) and low similarity (<0.8)

### 4. Full Pipeline (Complete Implementation)

```bash
python3 fullPipeline.py
```

Complete pipeline including all stages:

- Shingling
- Jaccard similarity (baseline)
- MinHash signature computation
- Signature similarity comparison
- LSH for efficient candidate detection

**Output:**

- Comparison of all three methods
- Summary of similar pairs found by each method
- Top 5 most similar pairs

### 5. Scalability Test ⭐

```bash
python3 scalabilityTest.py
```

**This is the main test for the assignment requirement!**

Tests execution time vs dataset size:

- Tests with 5, 10, and more documents (up to available)
- Measures execution time for each dataset size
- Calculates time per document pair
- Analyzes scaling behavior
- Uses similarity threshold s = 0.8

**Output:**

- Execution time for each dataset size
- Number of document pairs processed
- Time per pair analysis
- Scaling analysis (linear/better/worse)

## 🔧 Customization

### Change Parameters

**In `documentSimilarity.py` or `fullPipeline.py`:**

```python
base_path = "Resources/Dataset"      # Dataset path
num_files = 10                        # Number of files to process
k = 9                                 # Shingle size
similarity_threshold = 0.8            # Similarity threshold
num_hash_functions = 100              # Number of hash functions for MinHashing
```

### Generate Larger Test Dataset

To test scalability with 1000+ files:

```bash
# Generate 1000 test files
python3 generateTestDataset.py

# Update scalabilityTest.py line 135:
# base_path = "Resources/Dataset_Large"
```

## 📊 How It Works

### Stage 1: Shingling

1. Normalizes text (lowercase, removes punctuation)
2. Creates k-shingles (k-character subsequences)
3. Hashes each shingle to an integer using MD5
4. Returns a set of hashed shingles

### Stage 2: MinHashing

1. Generates n random hash functions: `(a*x + b) mod prime`
2. For each hash function, finds minimum hash value across all shingles
3. Creates a signature vector of length n
4. Signatures approximate Jaccard similarity efficiently

### Stage 3: LSH (Locality-Sensitive Hashing)

1. Divides each signature into bands
2. Hashes each band to create buckets
3. Documents in the same bucket are candidate pairs
4. Verifies candidates by computing actual similarity
5. Significantly reduces number of comparisons needed

### Stage 4: Similarity Computation

- **Jaccard Similarity**: `|A ∩ B| / |A ∪ B|` (direct comparison of shingle sets)
- **Signature Similarity**: Fraction of matching components in minHash signatures
- Both return similarity scores between 0.0 and 1.0

### Parallel Processing

- Uses Apache Spark RDDs for parallel document processing
- Efficient for large document collections
- Better scaling than sequential processing

## 📝 Example Usage

### Basic Usage

```python
from shingling import Shingling
from compareSets import CompareSets

# Create shingles
shingler = Shingling(k=9)
shingles1 = shingler.create_shingles("Document 1 text...")
shingles2 = shingler.create_shingles("Document 2 text...")

# Compute Jaccard similarity
similarity = CompareSets.jaccard_similarity(shingles1, shingles2)
print(f"Similarity: {similarity:.4f}")
```

### With MinHashing

```python
from minHashing import MinHashing
from compareSignatures import CompareSignatures

# Create minHash signatures
minhasher = MinHashing(n=100)
sig1 = minhasher.compute_signature(shingles1)
sig2 = minhasher.compute_signature(shingles2)

# Compare signatures
sig_sim = CompareSignatures.similarity(sig1, sig2)
print(f"Signature similarity: {sig_sim:.4f}")
```

### With LSH

```python
from lsh import LSH

# Find candidate pairs using LSH
lsh = LSH(threshold=0.8)
signatures = [sig1, sig2, sig3, ...]  # All signatures
candidate_pairs = lsh.find_candidate_pairs(signatures)

# Get verified similar pairs
similar_pairs = lsh.find_similar_pairs(signatures, verify=True)
```

## 🧪 Testing

### Test Sequence

1. **Unit Tests**: `python3 testClasses.py`
2. **Basic Analysis**: `python3 documentSimilarity.py`
3. **Full Pipeline**: `python3 fullPipeline.py`
4. **Scalability Test**: `python3 scalabilityTest.py` ⭐

### Expected Results

**Scalability Test Output:**

```
Files      Pairs      Time (s)        Time/Pair (ms)
5          10         1.83            183.19
10         45         1.52            33.69
12         66         1.48            22.41

Scaling Analysis:
From 5 to 10 files:
  Files increase: 2.00x
  Pairs increase: 4.50x
  Time increase: 0.83x
  ✓ Better than linear scaling
```

## 📈 Scalability Analysis

The scalability test measures:

- **Execution time** vs **dataset size**
- **Time per document pair** to understand efficiency
- **Scaling behavior** (linear, better, or worse)

**Assignment Requirement:**

- ✅ Tests with corpus of 5-10+ documents
- ✅ Uses similarity threshold s = 0.8
- ✅ Measures execution time vs dataset size
- ✅ Documents are similar if Jaccard similarity ≥ 0.8

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pyspark'"

```bash
pip3 install pyspark
```

### Java errors

```bash
java -version  # Check if Java is installed
# Install if needed: brew install openjdk (macOS)
```

### "Folder not found" error

```bash
# Make sure you're in the project directory
cd /path/to/ID2222-Data-Minig

# Check if dataset exists
ls Resources/Dataset/
```

### Spark warnings

- Warnings about native libraries are normal and can be ignored
- Spark will use built-in Java classes if native libraries aren't available

## 📚 Documentation

- **TEST_COMMANDS.md** - Detailed list of all test commands
- **DATASET_RECOMMENDATIONS.md** - Suggestions for datasets and testing strategies

## ✨ Features

- ✅ All required classes implemented
- ✅ Optional LSH implementation (bonus points)
- ✅ Scalability testing with execution time measurement
- ✅ Parallel processing with Apache Spark
- ✅ Comprehensive test suite
- ✅ Easy to customize parameters
- ✅ Well-documented code

## 📄 License

This project is for educational purposes as part of the ID2222 Data Mining course.

## 👤 Author

Student project for ID2222 Data Mining course at KTH.

---

**Note**: This implementation satisfies all assignment requirements including scalability testing with execution time measurement for datasets of 5-10+ documents using similarity threshold s = 0.8.
