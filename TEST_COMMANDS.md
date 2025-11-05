# Test Commands for Document Similarity Project

## Quick Test Commands

### 1. Test All Classes (Simple Unit Tests)

```bash
python3 testClasses.py
```

Tests all individual classes (Shingling, MinHashing, CompareSignatures, LSH) with sample data.

---

### 2. Simple Demo (Hardcoded Documents)

```bash
python3 main.py
```

Simple demo with 3 hardcoded sample documents using Spark.

---

### 3. Document Similarity Analysis (Basic)

```bash
python3 documentSimilarity.py
```

- Loads documents from `Resources/Dataset`
- Uses Shingling and Jaccard similarity
- Processes 10 random files by default
- Shows high/low similarity pairs

**Customize:**

```bash
# Edit documentSimilarity.py to change:
# - base_path = "Resources/Dataset"
# - num_files = 10
# - k = 9 (shingle size)
```

---

### 4. Full Pipeline (Complete: Shingling → MinHashing → LSH)

```bash
python3 fullPipeline.py
```

- Complete pipeline with all stages
- Uses Shingling, MinHashing, CompareSignatures, and LSH
- Processes 10 files by default
- Shows comparison between Jaccard, MinHash, and LSH results

**Customize:**

```bash
# Edit fullPipeline.py to change:
# - base_path = "Resources/Dataset"
# - num_files = 10
# - k = 9
# - num_hash_functions = 100
# - similarity_threshold = 0.8
```

---

## Advanced Testing

### Test with Different Parameters

#### Test with more documents:

Edit the files and change `num_files` parameter:

```python
# In documentSimilarity.py or fullPipeline.py
app.run(base_path, num_files=20)  # Process 20 files instead of 10
```

#### Test with different shingle size:

```python
app = DocumentSimilarity(k=5)  # Use k=5 instead of k=9
```

#### Test with different similarity threshold:

```python
pipeline = FullPipeline(k=9, num_hash_functions=100, similarity_threshold=0.7)
```

---

## Testing Individual Classes

### Test Shingling Only:

```python
python3 -c "
from shingling import Shingling
shingler = Shingling(k=9)
text = 'Hello world this is a test'
shingles = shingler.create_shingles(text)
print(f'Shingles: {len(shingles)}')
print(f'Sample: {list(shingles)[:5]}')
"
```

### Test MinHashing Only:

```python
python3 -c "
from shingling import Shingling
from minHashing import MinHashing
shingler = Shingling(k=9)
minhasher = MinHashing(n=20)
text = 'Hello world this is a test'
shingles = shingler.create_shingles(text)
sig = minhasher.compute_signature(shingles)
print(f'Signature: {sig[:10]}')
"
```

---

## Performance Testing

### Time the execution:

```bash
time python3 documentSimilarity.py
time python3 fullPipeline.py
```

### Compare performance with different number of files:

```bash
# Test with 5 files
python3 -c "
from documentSimilarity import DocumentSimilarity
import time
app = DocumentSimilarity(k=9)
start = time.time()
app.run('Resources/Dataset', num_files=5)
print(f'Time: {time.time() - start:.2f} seconds')
"

# Test with 10 files
python3 -c "
from documentSimilarity import DocumentSimilarity
import time
app = DocumentSimilarity(k=9)
start = time.time()
app.run('Resources/Dataset', num_files=10)
print(f'Time: {time.time() - start:.2f} seconds')
"
```

---

## Recommended Test Sequence

1. **Start with simple tests:**

   ```bash
   python3 testClasses.py
   ```

2. **Test basic functionality:**

   ```bash
   python3 documentSimilarity.py
   ```

3. **Test full pipeline:**

   ```bash
   python3 fullPipeline.py
   ```

4. **Compare results** - check that:
   - Jaccard similarity finds similar pairs
   - MinHash signatures approximate Jaccard similarity
   - LSH finds the same candidate pairs efficiently

---

## Troubleshooting

### If you get "ModuleNotFoundError: No module named 'pyspark'":

```bash
pip3 install pyspark
```

### If you get Java errors:

```bash
# Check Java version
java -version

# Install Java if needed (macOS):
brew install openjdk
```

### If you get "Folder not found" error:

```bash
# Make sure you're in the project directory
cd /Users/hieuvutongminh/KTH/2025.1.2/DataMining/homework/ID2222-Data-Minig

# Check if dataset exists
ls Resources/Dataset/
```
