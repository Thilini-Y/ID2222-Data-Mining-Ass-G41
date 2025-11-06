import itertools
import os
import time
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from compareSets import CompareSets
from compareSignature import CompareSignatures
from lsh import LSH
from minHashing import MinHashing
from shingling import Shingling


class PerformanceTest:
    """
    Evaluate scalability and accuracy of document similarity implementation.
    Runs the pipeline (Shingling → Jaccard → MinHash → LSH) and plots results.
    """

    def __init__(self, base_path, k=9, num_perm=1000):
        self.base_path = base_path
        self.k = k
        self.num_perm = num_perm

        # Initialize a shared Spark session
        self.spark = SparkSession.builder \
            .appName('PerformanceEvaluation') \
            .master('local[*]') \
            .getOrCreate()
        self.spark_context = self.spark.sparkContext

        # Initialize helper classes (these don’t reference SparkContext)
        self.shingler = Shingling(k)
        self.minhasher = MinHashing(num_perm=num_perm)

    # -------------------------------------------------------
    # Core processing pipeline
    # -------------------------------------------------------
    def load_documents(self):
        """Load all text files in the folder."""
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Folder not found: {self.base_path}")

        files = [
            f for f in os.listdir(self.base_path)
            if os.path.isfile(os.path.join(self.base_path, f)) and not f.startswith('.')
        ]
        if not files:
            raise ValueError(f"No files found in folder: {self.base_path}")

        selected_files = [(f, os.path.join(self.base_path, f)) for f in files]
        files_rdd = self.spark_context.parallelize(selected_files)
        # Read all files as (filename, text)
        return files_rdd.map(lambda x: (x[0], open(x[1], "r", errors="ignore").read()))

    def process_documents(self, limit_docs=None):
        """Run the full pipeline and return metrics for evaluation."""
        start_time = time.time()

        docs_rdd = self.load_documents()
        if limit_docs:
            docs_rdd = self.spark_context.parallelize(docs_rdd.take(limit_docs))

        # --- Step 1: Shingling ---
        shingler = self.shingler  # local variable to avoid SparkContext serialization
        shingled_rdd = docs_rdd.map(lambda x: (x[0], shingler.create_shingles(x[1])))
        documents = shingled_rdd.collect()
        num_docs = len(documents)

        # --- Step 2: True Jaccard Similarity ---
        jaccard_results = []
        for (f1, s1), (f2, s2) in itertools.combinations(documents, 2):
            sim = CompareSets.jaccard_similarity(s1, s2)
            jaccard_results.append(((f1, f2), sim))

        # --- Step 3: MinHash Signatures ---
        signatures_rdd = self.minhasher.compute_signatures_rdd(shingled_rdd)
        doc_signatures = dict(signatures_rdd.collect())

        # --- Step 4: Estimated Similarity using signatures ---
        estimated_results = []
        for (f1, sig1), (f2, sig2) in itertools.combinations(doc_signatures.items(), 2):
            sim = CompareSignatures.similarity(sig1, sig2)
            estimated_results.append(((f1, f2), sim))

        # --- Step 5: LSH Candidate Pairs ---
        lsh = LSH(bands=50, rows_per_band=20)
        candidates = lsh.find_candidates(doc_signatures)

        end_time = time.time()

        return {
            "num_docs": num_docs,
            "execution_time": end_time - start_time,
            "jaccard_results": jaccard_results,
            "estimated_results": estimated_results,
            "num_candidates": len(candidates)
        }

    # -------------------------------------------------------
    # Evaluation & Graphs
    # -------------------------------------------------------
    def run(self):
        """Run scalability and accuracy evaluation."""
        doc_counts = [3, 5, 7, 10]  # Adjust according to your dataset
        all_results = []

        for n in doc_counts:
            print(f"\nRunning with {n} documents...")
            result = self.process_documents(limit_docs=n)
            all_results.append(result)

        self.spark.stop()

        # ---- Plot results ----
        self.plot_execution_time(all_results)
        self.plot_jaccard_vs_estimated(all_results[-1])
        self.plot_candidates_vs_docs(all_results)

    def plot_execution_time(self, results):
        """Plot execution time vs number of documents."""
        plt.figure()
        plt.plot(
            [r["num_docs"] for r in results],
            [r["execution_time"] for r in results],
            marker='o', color='blue'
        )
        plt.title("Execution Time vs Number of Documents")
        plt.xlabel("Number of Documents")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(True)
        plt.show()

    def plot_jaccard_vs_estimated(self, result):
        """Scatter plot of True vs Estimated Similarity."""
        jaccard_dict = {pair: val for pair, val in result["jaccard_results"]}
        estimated_dict = {pair: val for pair, val in result["estimated_results"]}
        common_pairs = set(jaccard_dict.keys()) & set(estimated_dict.keys())

        true_vals = [jaccard_dict[p] for p in common_pairs]
        est_vals = [estimated_dict[p] for p in common_pairs]

        plt.figure()
        plt.scatter(true_vals, est_vals, color='purple')
        plt.plot([0, 1], [0, 1], '--', color='red')
        plt.title("True Jaccard vs Estimated MinHash Similarity")
        plt.xlabel("True Jaccard Similarity")
        plt.ylabel("Estimated MinHash Similarity")
        plt.grid(True)
        plt.show()

    def plot_candidates_vs_docs(self, results):
        """Plot LSH candidate pairs vs number of documents."""
        plt.figure()
        plt.bar(
            [r["num_docs"] for r in results],
            [r["num_candidates"] for r in results],
            color='orange'
        )
        plt.title("Number of LSH Candidate Pairs vs Dataset Size")
        plt.xlabel("Number of Documents")
        plt.ylabel("Candidate Pairs Found")
        plt.grid(True)
        plt.show()


# -------------------------------------------------------
# Run the performance evaluation
# -------------------------------------------------------
if __name__ == "__main__":
    base_path = "Resources/Dataset"  # Update path if needed
    tester = PerformanceTest(base_path=base_path, k=9, num_perm=1000)
    tester.run()
