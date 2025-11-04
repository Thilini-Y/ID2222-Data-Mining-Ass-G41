# Dataset Recommendations for Document Similarity Assignment

## Your Current Dataset ✅

**Status**: Good to use! You have USENET newsgroup posts which are perfect for this assignment.

**Current Files**: 12 documents in `Resources/Dataset/`

- These appear to be from the 20 Newsgroups dataset
- Some are duplicates/variants (good for testing similarity detection)
- Text-based content suitable for shingling

## Recommended Dataset Sources

### 1. **UCI Machine Learning Repository** (As mentioned in assignment)

- **20 Newsgroups Dataset**: Classic text classification dataset
  - Download: https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
  - Contains ~20,000 newsgroup posts across 20 topics
  - Perfect for similarity testing - documents from same topic should be similar
- **Reuters-21578**: News articles
  - Download: https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection
  - Contains news articles with categories
- **Spam Assassin**: Email dataset
  - Download: https://archive.ics.uci.edu/ml/datasets/Spambase (or original emails)
  - Good for testing email similarity

### 2. **20 Newsgroups (Scikit-learn)**

```python
from sklearn.datasets import fetch_20newsgroups
# Easy to load and use in Python
```

### 3. **Web Scraped Content**

- Wikipedia articles on related topics
- News articles from same publication
- Blog posts
- Product descriptions

### 4. **Create Your Own Test Set**

You can create a controlled test set with:

- **5-10 original documents** on different topics
- **2-3 duplicates** of some documents (for testing similarity threshold)
- **2-3 modified versions** (80-90% similar content)
- **Varying lengths**: Mix of short (100 words) and long (1000+ words)

## Dataset Characteristics for Best Results

### ✅ Good Characteristics:

1. **Plain text format** (no complex formatting)
2. **Minimum 200-500 words per document** (for meaningful shingles)
3. **Some known similar pairs** (to validate your algorithm)
4. **Mix of topics** (some similar, some different)
5. **Real-world content** (not synthetic)

### ❌ Avoid:

- Binary files (PDFs, images) - need text extraction first
- Very short documents (< 50 words) - not enough shingles
- Highly structured data (JSON, XML) - need preprocessing
- Non-English text (unless you adjust normalization)

## Recommended Dataset Structure

```
Resources/Dataset/
├── original_1.txt          # Original document 1
├── original_2.txt          # Original document 2
├── original_3.txt          # Original document 3
├── duplicate_1.txt         # Exact copy of original_1 (similarity = 1.0)
├── modified_1.txt          # 80% similar to original_1
├── similar_topic_1.txt     # Similar topic to original_1
├── different_topic_1.txt   # Different topic
└── ...
```

## Testing Strategy

1. **Similarity Threshold Testing**: Use threshold = 0.8

   - Documents with similarity ≥ 0.8 should be detected
   - Your current dataset has `*_copy_updated` files - test if these show high similarity

2. **Scalability Testing**:

   - Start with 5 documents
   - Test with 10, 20, 50, 100 documents
   - Measure execution time vs. dataset size

3. **Known Similar Pairs**:
   - `49960` vs `49960_copy_updated` should have high similarity
   - `51121` vs `51121 copy` should have high similarity
   - Different numbered files likely have lower similarity

## Quick Dataset Validation

Check if your dataset is suitable:

```python
import os

def validate_dataset(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print(f"Total files: {len(files)}")

    for f in files[:5]:
        with open(os.path.join(path, f), 'r', errors='ignore') as file:
            content = file.read()
            print(f"{f}: {len(content)} characters, {len(content.split())} words")
```

## Suggested Dataset Size for Assignment

- **Minimum**: 5-10 documents (required by assignment)
- **Recommended**: 20-50 documents (better for scalability testing)
- **Optimal**: 50-200 documents (for comprehensive evaluation)

Your current 12 documents is a good starting point! You can expand it later if needed.
