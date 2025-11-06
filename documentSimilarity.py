import itertools
import os
import random
from pyspark.sql import SparkSession

from compareSets import CompareSets
from compareSignature import CompareSignatures
from lsh import LSH
from minHashing import MinHashing
from shingling import Shingling


class DocumentSimilarity:
    def __init__(self, k=9, num_perm=100):
        # Initialize shingling (k-shingles), MinHashing (num_perm permutations), and Spark
        self.k = k
        self.shingler = Shingling(k)
        self.minhasher = MinHashing(num_perm=num_perm)
        self.spark = (
            SparkSession.builder.appName("DocumentSimilarity")
            .master("local[*]")
            .getOrCreate()
        )
        self.spark_context = self.spark.sparkContext

    def load_documents(self, path, num_files=5):
        # Step 1: Load documents from folder
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        # Get all text files from folder
        files = [
            f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")
        ]

        if not files:
            raise ValueError(f"No files found in folder: {path}")

        # Randomly select subset of files
        selected_files = random.sample(files, min(num_files, len(files)))

        # Create (filename, filepath) pairs
        selected_files = [(f, os.path.join(path, f)) for f in selected_files]
        print(f"Folder: {path}")
        print(
            f"Files selected ({len(selected_files)}): {[f for f, _ in selected_files]}"
        )

        # Read files in parallel using Spark RDD
        files_rdd = self.spark_context.parallelize(selected_files)
        read_files_rdd = files_rdd.map(
            lambda x: (x[0], open(x[1], "r", errors="ignore").read())
        )
        return read_files_rdd

    def run(self, folder_path, num_files=5):
        # Step 1: Load documents
        documents_rdd = self.load_documents(folder_path, num_files)
        shingler = self.shingler

        # Step 2: Create shingles (k-grams) for each document
        shingled_rdd = documents_rdd.map(
            lambda x: (x[0], shingler.create_shingles(x[1]))
        )

        # Display sample shingles
        results = shingled_rdd.take(3)
        for file, shingles in results:
            print(f"\nFile: {file}")
            print(f"Number of shingles: {len(shingles)}")
            print("Sample hashed shingles (first 5):")
            for s in list(shingles)[:5]:
                print(f"  {s}")

        # Collect all shingled documents
        documents = shingled_rdd.collect()
        print(f"\nTotal documents processed: {len(documents)}")

        high_similarity = []
        low_similarity = []

        # Step 3: Compute exact Jaccard similarity (all pairs)
        print("\n**********Jaccard Similarity**********")
        for (file1, s1), (file2, s2) in itertools.combinations(documents, 2):
            jaccard_similarity = CompareSets.jaccard_similarity(s1, s2)
            print_data = (
                f"{file1} <-> {file2} Jaccard similarity = {jaccard_similarity:.4f}"
            )
            if jaccard_similarity >= 0.8:
                high_similarity.append(print_data)
            else:
                low_similarity.append(print_data)

        print("\n-----High Jaccard Similarity-----")
        for i in high_similarity:
            print(i)

        print("\n-----Low Jaccard Similarity-----")
        for j in low_similarity:
            print(j)

        # Step 4: Compute MinHash signatures (compressed representation)
        print("\n**********Computing Similarity using MinHash**********")
        signatures_rdd = self.minhasher.compute_signatures_rdd(shingled_rdd)
        doc_signatures = dict(signatures_rdd.collect())
        print(f"Generated signatures for total documents: {len(doc_signatures)}")

        # Step 5: Estimate similarity using MinHash signatures
        print("\n**********Compare Similarities using signatures**********")
        high_estimated = []
        low_estimated = []
        for (file_1, signature_1), (file_2, signature_2) in itertools.combinations(
            doc_signatures.items(), 2
        ):
            compared_similarity = CompareSignatures.similarity(signature_1, signature_2)
            print_data = f"{file_1} <-> {file_2}  = {compared_similarity:.4f}"
            if compared_similarity >= 0.8:
                high_estimated.append(print_data)
            else:
                low_estimated.append(print_data)

        print("\n----- High Estimated Similarity -----")
        for i in high_estimated:
            print(i)

        print("\n----- Low Estimated Similarity -----")
        for j in low_estimated:
            print(j)

        # Step 6: Find candidate pairs using LSH (efficient filtering)
        print("\n**********Finding Candidate Pairs using LSH**********")

        # Set similarity threshold
        similarity_threshold = 0.8

        # Find optimal bands and rows_per_band for target threshold
        num_perm = self.minhasher.num_perm
        optimal_bands, optimal_rows = LSH.find_optimal_bands_rows(
            num_perm, similarity_threshold
        )
        print(
            f"Optimal LSH config for threshold {similarity_threshold}: bands={optimal_bands}, rows_per_band={optimal_rows}"
        )
        print(
            f"Expected threshold: {LSH.calculate_threshold(optimal_bands, optimal_rows):.4f}"
        )

        # Initialize LSH and find candidate pairs
        lsh = LSH(bands=optimal_bands, rows_per_band=optimal_rows)
        candidates = lsh.find_candidates(doc_signatures)

        # Display results
        if not candidates:
            print("No high-similarity candidate pairs detected by LSH.")
        else:
            print(f"Total LSH candidate pairs: {len(candidates)}")
            for doc1, doc2 in candidates:
                print(f"Candidate pair: {doc1} <-> {doc2}")

        self.spark.stop()


if __name__ == "__main__":
    # Initialize and run document similarity pipeline
    base_path = "Resources/Dataset"
    app = DocumentSimilarity(
        k=9, num_perm=1000
    )  # k=9 shingles, 1000 MinHash permutations
    app.run(base_path, num_files=10)
