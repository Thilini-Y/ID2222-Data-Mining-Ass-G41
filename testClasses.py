# Test script for all classes
from shingling import Shingling
from compareSets import CompareSets
from minHashing import MinHashing
from compareSignatures import CompareSignatures
from lsh import LSH


def test_shingling():
    # Test Shingling class
    print("=" * 60)
    print("Testing Shingling")
    print("=" * 60)

    shingler = Shingling(k=5)
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The quick brown fox jumps over the lazy dog"  # identical
    text3 = "The fast brown fox leaps over the lazy hound"  # similar but different
    text4 = "Python is a great programming language"  # completely different

    shingles1 = shingler.create_shingles(text1)
    shingles2 = shingler.create_shingles(text2)
    shingles3 = shingler.create_shingles(text3)
    shingles4 = shingler.create_shingles(text4)

    print(f"Text 1: '{text1}'")
    print(f"  Shingles: {len(shingles1)} unique shingles")
    print(f"  Sample: {list(shingles1)[:5]}")

    print(f"\nText 2: '{text2}' (identical)")
    print(f"  Shingles: {len(shingles2)} unique shingles")

    print(f"\nText 3: '{text3}' (similar)")
    print(f"  Shingles: {len(shingles3)} unique shingles")

    print(f"\nText 4: '{text4}' (different)")
    print(f"  Shingles: {len(shingles4)} unique shingles")

    # Test similarity
    sim12 = CompareSets.jaccard_similarity(shingles1, shingles2)
    sim13 = CompareSets.jaccard_similarity(shingles1, shingles3)
    sim14 = CompareSets.jaccard_similarity(shingles1, shingles4)

    print(f"\nJaccard similarity:")
    print(f"  Text1 vs Text2 (identical): {sim12:.4f}")
    print(f"  Text1 vs Text3 (similar): {sim13:.4f}")
    print(f"  Text1 vs Text4 (different): {sim14:.4f}")
    print()


def test_minhashing():
    # Test MinHashing class
    print("=" * 60)
    print("Testing MinHashing")
    print("=" * 60)

    shingler = Shingling(k=5)
    minhasher = MinHashing(n=20, seed=42)

    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "The quick brown fox jumps over the lazy dog"  # identical
    text3 = "The fast brown fox leaps over the lazy hound"  # similar
    text4 = "Python is a great programming language"  # different

    shingles1 = shingler.create_shingles(text1)
    shingles2 = shingler.create_shingles(text2)
    shingles3 = shingler.create_shingles(text3)
    shingles4 = shingler.create_shingles(text4)

    sig1 = minhasher.compute_signature(shingles1)
    sig2 = minhasher.compute_signature(shingles2)
    sig3 = minhasher.compute_signature(shingles3)
    sig4 = minhasher.compute_signature(shingles4)

    print(f"Signature 1: {sig1[:10]}... (length: {len(sig1)})")
    print(f"Signature 2: {sig2[:10]}... (length: {len(sig2)})")
    print(f"Signature 3: {sig3[:10]}... (length: {len(sig3)})")
    print(f"Signature 4: {sig4[:10]}... (length: {len(sig4)})")

    # Compare signatures
    sim12 = CompareSignatures.similarity(sig1, sig2)
    sim13 = CompareSignatures.similarity(sig1, sig3)
    sim14 = CompareSignatures.similarity(sig1, sig4)

    print(f"\nSignature similarity:")
    print(f"  Sig1 vs Sig2 (identical): {sim12:.4f}")
    print(f"  Sig1 vs Sig3 (similar): {sim13:.4f}")
    print(f"  Sig1 vs Sig4 (different): {sim14:.4f}")
    print()


def test_lsh():
    # Test LSH class
    print("=" * 60)
    print("Testing LSH")
    print("=" * 60)

    shingler = Shingling(k=5)
    minhasher = MinHashing(n=100, seed=42)  # need enough hash functions for LSH
    lsh = LSH(threshold=0.8)

    # Create several documents
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",  # Duplicate
        "The fast brown fox leaps over the lazy hound",  # Similar
        "Python is a great programming language",
        "Java is also a great programming language",  # Similar to Python doc
        "The weather today is sunny and warm",
    ]

    doc_names = [f"doc_{i}" for i in range(len(texts))]

    # Create shingles and signatures
    shingles_list = [shingler.create_shingles(text) for text in texts]
    signatures = minhasher.compute_signature_batch(shingles_list)

    print(f"Documents: {len(texts)}")
    print(f"Signatures: {len(signatures)} of length {len(signatures[0])}")

    # Find candidate pairs using LSH
    candidate_pairs = lsh.find_candidate_pairs(signatures, doc_names)
    print(f"\nLSH candidate pairs: {len(candidate_pairs)}")
    for doc1_idx, doc2_idx in candidate_pairs:
        print(f"  {doc_names[doc1_idx]} <-> {doc_names[doc2_idx]}")

    # Find similar pairs with verification
    similar_pairs = lsh.find_similar_pairs(signatures, doc_names, verify=True)
    print(f"\nLSH similar pairs (>= 0.8): {len(similar_pairs)}")
    for doc1_idx, doc2_idx, sim in similar_pairs:
        print(f"  {doc_names[doc1_idx]} <-> {doc_names[doc2_idx]}: {sim:.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING ALL CLASSES")
    print("=" * 60 + "\n")

    try:
        test_shingling()
        test_minhashing()
        test_lsh()
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
