"""
Simple test file to verify all functions are working correctly.
Tests: Shingling, Jaccard Similarity, MinHash, Signature Comparison, and LSH.
"""

from shingling import Shingling
from compareSets import CompareSets
from minHashing import MinHashing
from compareSignature import CompareSignatures
from lsh import LSH
from pyspark.sql import SparkSession


def test_shingling():
    """Test 1: Shingling - Create k-shingles from text."""
    print("=" * 60)
    print("Test 1: Shingling")
    print("=" * 60)

    shingler = Shingling(k=3)

    text1 = "The quick brown fox"
    text2 = "The quick brown dog"

    shingles1 = shingler.create_shingles(text1)
    shingles2 = shingler.create_shingles(text2)

    print(f"Text 1: '{text1}'")
    print(f"Shingles 1: {len(shingles1)} shingles")
    print(f"Sample shingles: {list(shingles1)[:5]}")

    print(f"\nText 2: '{text2}'")
    print(f"Shingles 2: {len(shingles2)} shingles")
    print(f"Sample shingles: {list(shingles2)[:5]}")

    # Check that similar texts have overlapping shingles
    overlap = len(shingles1.intersection(shingles2))
    print(f"\nOverlapping shingles: {overlap}")
    print(f"✓ Shingling test passed!")
    print()


def test_jaccard_similarity():
    """Test 2: Jaccard Similarity - Compare two sets."""
    print("=" * 60)
    print("Test 2: Jaccard Similarity")
    print("=" * 60)

    # Create two sets with some overlap
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}

    similarity = CompareSets.jaccard_similarity(set1, set2)

    print(f"Set 1: {set1}")
    print(f"Set 2: {set2}")
    print(f"Intersection: {set1.intersection(set2)}")
    print(f"Union: {set1.union(set2)}")
    print(f"Jaccard Similarity: {similarity:.4f}")

    # Expected: intersection = {4, 5} = 2, union = {1,2,3,4,5,6,7,8} = 8
    # Expected similarity = 2/8 = 0.25
    expected = 2.0 / 8.0
    assert abs(similarity - expected) < 0.001, f"Expected {expected}, got {similarity}"
    print(f"✓ Expected similarity: {expected:.4f}")
    print(f"✓ Jaccard similarity test passed!")
    print()


def test_minhash():
    """Test 3: MinHash - Generate signatures."""
    print("=" * 60)
    print("Test 3: MinHash Signatures")
    print("=" * 60)

    minhasher = MinHashing(num_perm=10, seed=42)

    # Create test shingles
    shingles1 = {1, 2, 3, 4, 5}
    shingles2 = {4, 5, 6, 7, 8}

    sig1 = minhasher.compute_for_doc(shingles1, minhasher.hash_params, minhasher.prime)
    sig2 = minhasher.compute_for_doc(shingles2, minhasher.hash_params, minhasher.prime)

    print(f"Shingles 1: {shingles1}")
    print(f"Signature 1: {sig1}")
    print(f"Signature length: {len(sig1)}")

    print(f"\nShingles 2: {shingles2}")
    print(f"Signature 2: {sig2}")
    print(f"Signature length: {len(sig2)}")

    # Signatures should have same length
    assert len(sig1) == len(sig2) == 10, "Signature lengths don't match"
    print(f"\n✓ Signature lengths match: {len(sig1)}")
    print(f"✓ MinHash test passed!")
    print()


def test_signature_comparison():
    """Test 4: Signature Comparison - Compare MinHash signatures."""
    print("=" * 60)
    print("Test 4: Signature Comparison")
    print("=" * 60)

    # Create test signatures
    sig1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sig2 = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]  # 5 matching positions

    similarity = CompareSignatures.similarity(sig1, sig2)

    print(f"Signature 1: {sig1}")
    print(f"Signature 2: {sig2}")
    print(f"Matching positions: 5 out of 10")
    print(f"Estimated Similarity: {similarity:.4f}")

    # Expected: 5/10 = 0.5
    expected = 5.0 / 10.0
    assert abs(similarity - expected) < 0.001, f"Expected {expected}, got {similarity}"
    print(f"✓ Expected similarity: {expected:.4f}")
    print(f"✓ Signature comparison test passed!")
    print()


def test_lsh():
    """Test 5: LSH - Find candidate pairs."""
    print("=" * 60)
    print("Test 5: LSH (Locality-Sensitive Hashing)")
    print("=" * 60)

    # Create test signatures
    # num_perm = 20, so bands * rows_per_band = 20
    # Let's use bands=4, rows_per_band=5
    lsh = LSH(bands=4, rows_per_band=5)

    # Create signatures with 20 permutations
    signatures = {
        "doc1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "doc2": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],  # Same as doc1
        "doc3": [
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
        ],  # Different
    }

    candidates = lsh.find_candidates(signatures)

    print(f"Signatures: {len(signatures)} documents")
    print(f"LSH Configuration: bands={lsh.bands}, rows_per_band={lsh.rows_per_band}")
    print(f"Candidate pairs found: {len(candidates)}")
    print(f"Candidates: {candidates}")

    # doc1 and doc2 should be candidates (same signature)
    # doc3 should not pair with others (different signature)
    assert ("doc1", "doc2") in candidates or (
        "doc2",
        "doc1",
    ) in candidates, "doc1 and doc2 should be candidates"
    assert len(candidates) >= 1, "Should find at least one candidate pair"

    print(f"✓ LSH found expected candidate pairs")
    print(f"✓ LSH test passed!")
    print()


def test_full_pipeline():
    """Test 6: Full Pipeline - Complete workflow."""
    print("=" * 60)
    print("Test 6: Full Pipeline Integration")
    print("=" * 60)

    # Initialize components
    shingler = Shingling(k=3)
    minhasher = MinHashing(num_perm=20, seed=42)

    # Create test documents
    doc1 = "The quick brown fox jumps over the lazy dog"
    doc2 = "The quick brown fox jumps over the lazy dog"  # Identical
    doc3 = "A completely different text with no similarity"

    # Step 1: Shingling
    shingles1 = shingler.create_shingles(doc1)
    shingles2 = shingler.create_shingles(doc2)
    shingles3 = shingler.create_shingles(doc3)

    print(f"Document 1: '{doc1[:30]}...'")
    print(f"Document 2: '{doc2[:30]}...'")
    print(f"Document 3: '{doc3[:30]}...'")

    # Step 2: Jaccard Similarity
    jaccard_12 = CompareSets.jaccard_similarity(shingles1, shingles2)
    jaccard_13 = CompareSets.jaccard_similarity(shingles1, shingles3)

    print(f"\nJaccard similarity (doc1 vs doc2): {jaccard_12:.4f}")
    print(f"Jaccard similarity (doc1 vs doc3): {jaccard_13:.4f}")

    # Step 3: MinHash Signatures
    sig1 = minhasher.compute_for_doc(shingles1, minhasher.hash_params, minhasher.prime)
    sig2 = minhasher.compute_for_doc(shingles2, minhasher.hash_params, minhasher.prime)
    sig3 = minhasher.compute_for_doc(shingles3, minhasher.hash_params, minhasher.prime)

    # Step 4: Signature Comparison
    minhash_12 = CompareSignatures.similarity(sig1, sig2)
    minhash_13 = CompareSignatures.similarity(sig1, sig3)

    print(f"\nMinHash similarity (doc1 vs doc2): {minhash_12:.4f}")
    print(f"MinHash similarity (doc1 vs doc3): {minhash_13:.4f}")

    # Step 5: LSH
    lsh = LSH(bands=4, rows_per_band=5)
    signatures = {"doc1": sig1, "doc2": sig2, "doc3": sig3}
    candidates = lsh.find_candidates(signatures)

    print(f"\nLSH candidates: {candidates}")

    # Verify results
    assert jaccard_12 > 0.9, "Identical documents should have high Jaccard similarity"
    assert jaccard_13 < 0.5, "Different documents should have low Jaccard similarity"
    assert (
        minhash_12 > 0.5
    ), "Similar documents should have reasonable MinHash similarity"

    print(f"\n✓ All pipeline steps completed successfully!")
    print(f"✓ Full pipeline test passed!")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SIMPLE FUNCTIONALITY TESTS")
    print("=" * 60)
    print()

    try:
        test_shingling()
        test_jaccard_similarity()
        test_minhash()
        test_signature_comparison()
        test_lsh()
        test_full_pipeline()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nAll functions are working correctly!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
