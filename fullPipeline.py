# Complete pipeline: Shingling -> MinHashing -> LSH
import os
import itertools
from pyspark.sql import SparkSession

from shingling import Shingling
from compareSets import CompareSets
from minHashing import MinHashing
from compareSignatures import CompareSignatures
from lsh import LSH


class FullPipeline:
    # Main pipeline that uses all stages

    def __init__(self, k=9, num_hash_functions=100, similarity_threshold=0.8):
        self.k = k
        self.num_hash_functions = num_hash_functions
        self.similarity_threshold = similarity_threshold

        # Initialize classes
        self.shingler = Shingling(k)
        self.minhasher = MinHashing(n=num_hash_functions)
        self.lsh = LSH(threshold=similarity_threshold)

        # Setup Spark
        self.spark = (
            SparkSession.builder.appName("FullPipeline")
            .master("local[*]")
            .getOrCreate()
        )
        self.spark_context = self.spark.sparkContext

    def load_documents(self, path, num_files=None):
        # Load documents from folder
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        files = [
            f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")
        ]

        if not files:
            raise ValueError(f"No files found in folder: {path}")

        if num_files:
            import random

            files = random.sample(files, min(num_files, len(files)))

        selected_files = [(f, os.path.join(path, f)) for f in files]
        print(f"Folder: {path}")
        print(
            f"Files selected ({len(selected_files)}): {[f for f, _ in selected_files]}"
        )

        # Read files with Spark
        files_rdd = self.spark_context.parallelize(selected_files)
        read_files_rdd = files_rdd.map(
            lambda x: (x[0], open(x[1], "r", errors="ignore").read())
        )
        return read_files_rdd

    def run_full_pipeline(self, folder_path, num_files=None, use_lsh=True):
        # Run all stages of the pipeline
        print("=" * 70)
        print("COMPLETE PIPELINE: Shingling -> MinHashing -> LSH")
        print("=" * 70)

        # Step 1: Load documents
        print("\n[Step 1] Loading documents...")
        documents_rdd = self.load_documents(folder_path, num_files)
        shingler = self.shingler

        # Step 2: Create shingles
        print("\n[Step 2] Creating shingles...")
        shingled_rdd = documents_rdd.map(
            lambda x: (x[0], shingler.create_shingles(x[1]))
        )

        # Show some results
        results = shingled_rdd.take(3)
        for file, shingles in results:
            print(f"  {file}: {len(shingles)} shingles")

        # Get all documents
        documents = shingled_rdd.collect()
        doc_names = [doc[0] for doc in documents]
        shingles_list = [doc[1] for doc in documents]

        print(f"\nTotal documents processed: {len(documents)}")

        # Step 3: Jaccard similarity
        print("\n[Step 3] Computing Jaccard similarity (baseline)...")
        jaccard_similarities = []
        for (file1, s1), (file2, s2) in itertools.combinations(documents, 2):
            jaccard_sim = CompareSets.jaccard_similarity(s1, s2)
            jaccard_similarities.append((file1, file2, jaccard_sim))

        high_jaccard = [
            (f1, f2, sim)
            for f1, f2, sim in jaccard_similarities
            if sim >= self.similarity_threshold
        ]
        print(
            f"  Pairs with Jaccard similarity >= {self.similarity_threshold}: {len(high_jaccard)}"
        )

        # Step 4: MinHash signatures
        print(
            f"\n[Step 4] Computing minHash signatures (n={self.num_hash_functions})..."
        )
        signatures = self.minhasher.compute_signature_batch(shingles_list)
        print(f"  Computed {len(signatures)} signatures of length {len(signatures[0])}")

        # Step 5: Compare signatures
        print("\n[Step 5] Comparing minHash signatures...")
        signature_similarities = []
        for sig1, sig2 in itertools.combinations(enumerate(signatures), 2):
            idx1, s1 = sig1
            idx2, s2 = sig2
            sig_sim = CompareSignatures.similarity(s1, s2)
            signature_similarities.append((doc_names[idx1], doc_names[idx2], sig_sim))

        high_signature = [
            (f1, f2, sim)
            for f1, f2, sim in signature_similarities
            if sim >= self.similarity_threshold
        ]
        print(
            f"  Pairs with signature similarity >= {self.similarity_threshold}: {len(high_signature)}"
        )

        # Step 6: LSH
        if use_lsh:
            print("\n[Step 6] Applying LSH...")
            similar_pairs = self.lsh.find_similar_pairs(
                signatures, doc_names, verify=True
            )
            print(f"  Candidate pairs found by LSH: {len(similar_pairs)}")

            if similar_pairs:
                print("\n  LSH Results (similarity >= threshold):")
                for doc1_idx, doc2_idx, sim in similar_pairs:
                    print(
                        f"    {doc_names[doc1_idx]} <-> {doc_names[doc2_idx]}: {sim:.4f}"
                    )
        else:
            similar_pairs = []

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            f"Total document pairs: {len(list(itertools.combinations(documents, 2)))}"
        )
        print(
            f"Jaccard similarity pairs (>= {self.similarity_threshold}): {len(high_jaccard)}"
        )
        print(
            f"MinHash signature similarity pairs (>= {self.similarity_threshold}): {len(high_signature)}"
        )
        if use_lsh:
            print(
                f"LSH candidate pairs (>= {self.similarity_threshold}): {len(similar_pairs)}"
            )

        # Top similar pairs
        print("\nTop 5 most similar pairs (by Jaccard similarity):")
        sorted_jaccard = sorted(jaccard_similarities, key=lambda x: x[2], reverse=True)[
            :5
        ]
        for f1, f2, sim in sorted_jaccard:
            print(f"  {f1} <-> {f2}: {sim:.4f}")

        self.spark.stop()

        return {
            "documents": documents,
            "signatures": signatures,
            "jaccard_similarities": jaccard_similarities,
            "signature_similarities": signature_similarities,
            "lsh_pairs": similar_pairs if use_lsh else [],
        }


if __name__ == "__main__":
    base_path = "Resources/Dataset"

    # Create and run pipeline
    pipeline = FullPipeline(k=9, num_hash_functions=100, similarity_threshold=0.8)
    results = pipeline.run_full_pipeline(base_path, num_files=10, use_lsh=True)
