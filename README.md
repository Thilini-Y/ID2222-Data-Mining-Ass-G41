# ID2222 Data Mining - Document Similarity Project

This project implements document similarity analysis using shingling and Jaccard similarity with Apache Spark.

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

### Customizing Parameters

You can modify the parameters in `documentSimilarity.py`:

```python
if __name__ == '__main__':
    base_path = "Resources/Dataset"  # Change dataset path
    app = DocumentSimilarity(k=9)     # Change shingle size (k)
    app.run(base_path, num_files=10)  # Change number of files to process
```

## Project Structure

- `shingling.py` - Implements the Shingling class for creating k-shingles
- `compareSets.py` - Implements Jaccard similarity calculation
- `documentSimilarity.py` - Main application for document similarity analysis
- `main.py` - Simple demo application
- `Resources/Dataset/` - Contains the document files to analyze

## How It Works

1. **Shingling**: Converts each document into a set of k-shingles (k-character subsequences)
2. **Hashing**: Shingles are hashed to integers using MD5
3. **Similarity**: Jaccard similarity is computed between all pairs of documents
4. **Parallel Processing**: Uses Spark RDDs to process documents in parallel
