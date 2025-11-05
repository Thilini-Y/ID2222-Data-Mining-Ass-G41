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
        self.spark = (
            SparkSession.builder.appName("PerformanceEvaluation")
            .master("local[*]")
            .getOrCreate()
        )
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
            f
            for f in os.listdir(self.base_path)
            if os.path.isfile(os.path.join(self.base_path, f)) and not f.startswith(".")
        ]
        if not files:
            raise ValueError(f"No files found in folder: {self.base_path}")

        selected_files = [(f, os.path.join(self.base_path, f)) for f in files]
        files_rdd = self.spark_context.parallelize(selected_files)
        # Read all files as (filename, text)
        return files_rdd.map(lambda x: (x[0], open(x[1], "r", errors="ignore").read()))

    def process_documents(
        self, limit_docs=None, similarity_threshold=0.8, use_lsh=True
    ):
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

        # Create a dictionary for quick lookup
        doc_dict = {f: s for f, s in documents}

        # --- Step 2: MinHash Signatures ---
        signatures_rdd = self.minhasher.compute_signatures_rdd(shingled_rdd)
        doc_signatures = dict(signatures_rdd.collect())

        jaccard_results = []
        estimated_results = []
        similar_pairs = []
        candidates = set()

        if use_lsh:
            # --- WITH LSH: Only compute similarities for candidate pairs ---
            # Step 3: Find candidate pairs using LSH
            lsh = LSH(bands=50, rows_per_band=20)
            candidates = lsh.find_candidates(doc_signatures)

            # Step 4: Compute similarities only for candidate pairs
            for f1, f2 in candidates:
                # Jaccard similarity
                s1, s2 = doc_dict[f1], doc_dict[f2]
                jaccard_sim = CompareSets.jaccard_similarity(s1, s2)
                jaccard_results.append(((f1, f2), jaccard_sim))

                # MinHash similarity
                sig1, sig2 = doc_signatures[f1], doc_signatures[f2]
                minhash_sim = CompareSignatures.similarity(sig1, sig2)
                estimated_results.append(((f1, f2), minhash_sim))

                # Check if similar
                if jaccard_sim >= similarity_threshold:
                    similar_pairs.append(((f1, f2), jaccard_sim))
        else:
            # --- WITHOUT LSH: Compute all pairwise similarities ---
            # Step 3: True Jaccard Similarity for all pairs
            for (f1, s1), (f2, s2) in itertools.combinations(documents, 2):
                sim = CompareSets.jaccard_similarity(s1, s2)
                jaccard_results.append(((f1, f2), sim))
                if sim >= similarity_threshold:
                    similar_pairs.append(((f1, f2), sim))

            # Step 4: Estimated Similarity using signatures for all pairs
            for (f1, sig1), (f2, sig2) in itertools.combinations(
                doc_signatures.items(), 2
            ):
                sim = CompareSignatures.similarity(sig1, sig2)
                estimated_results.append(((f1, f2), sim))

        end_time = time.time()

        return {
            "num_docs": num_docs,
            "execution_time": end_time - start_time,
            "jaccard_results": jaccard_results,
            "estimated_results": estimated_results,
            "similar_pairs": similar_pairs,
            "num_similar_pairs": len(similar_pairs),
            "num_candidates": len(candidates),
            "similarity_threshold": similarity_threshold,
            "use_lsh": use_lsh,
        }

    # -------------------------------------------------------
    # Evaluation & Graphs
    # -------------------------------------------------------
    def run(self, similarity_threshold=0.8):
        """Run scalability and accuracy evaluation with 5-1000 documents."""
        # Test with increasing document counts from 5 to 1000
        doc_counts = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        all_results = []

        print(f"\n{'='*60}")
        print(f"Testing scalability with similarity threshold: {similarity_threshold}")
        print(
            f"Two documents are similar if Jaccard similarity >= {similarity_threshold}"
        )
        print(f"{'='*60}\n")

        # Warmup run to eliminate Spark initialization overhead
        print("Performing warmup run to eliminate initialization overhead...")
        _ = self.process_documents(
            limit_docs=5, similarity_threshold=similarity_threshold, use_lsh=True
        )
        print("Warmup complete. Starting actual measurements...\n")

        all_results_with_lsh = []
        all_results_without_lsh = []

        for n in doc_counts:
            print(f"Running with {n} documents...")

            # Run with LSH
            print(f"  Running WITH LSH...")
            result_with_lsh = self.process_documents(
                limit_docs=n, similarity_threshold=similarity_threshold, use_lsh=True
            )
            all_results_with_lsh.append(result_with_lsh)

            # Run without LSH
            print(f"  Running WITHOUT LSH...")
            result_without_lsh = self.process_documents(
                limit_docs=n, similarity_threshold=similarity_threshold, use_lsh=False
            )
            all_results_without_lsh.append(result_without_lsh)

            print(
                f"  Execution time WITH LSH: {result_with_lsh['execution_time']:.4f} seconds"
            )
            print(
                f"  Execution time WITHOUT LSH: {result_without_lsh['execution_time']:.4f} seconds"
            )
            print(
                f"  Similar pairs found (Jaccard >= {similarity_threshold}): {result_with_lsh['num_similar_pairs']}"
            )
            print(f"  Total pairs: {len(result_with_lsh['jaccard_results'])}")
            print()

        # Print summary before plotting
        print(f"\n{'='*60}")
        print("Summary of Results:")
        print(f"{'='*60}")
        print("WITH LSH:")
        for r in all_results_with_lsh:
            print(
                f"  {r['num_docs']} docs: {r['execution_time']:.4f}s, "
                f"{r['num_similar_pairs']} similar pairs, "
                f"{r['num_candidates']} LSH candidates"
            )
        print("\nWITHOUT LSH:")
        for r in all_results_without_lsh:
            print(
                f"  {r['num_docs']} docs: {r['execution_time']:.4f}s, "
                f"{r['num_similar_pairs']} similar pairs"
            )
        print(f"{'='*60}\n")

        self.spark.stop()

        # ---- Plot results ----
        self.plot_execution_time_comparison(
            all_results_with_lsh, all_results_without_lsh
        )
        self.plot_jaccard_vs_estimated(all_results_with_lsh[-1])
        self.plot_candidates_vs_docs(all_results_with_lsh)

    def plot_execution_time_comparison(self, results_with_lsh, results_without_lsh):
        """
        Plot execution time comparison with and without LSH.
        This is the main scalability graph required by the assignment.
        """
        plt.figure(figsize=(12, 7))
        doc_counts = [r["num_docs"] for r in results_with_lsh]
        exec_times_with_lsh = [r["execution_time"] for r in results_with_lsh]
        exec_times_without_lsh = [r["execution_time"] for r in results_without_lsh]

        plt.plot(
            doc_counts,
            exec_times_with_lsh,
            marker="o",
            markersize=8,
            linewidth=2,
            color="blue",
            label="With LSH",
        )

        plt.plot(
            doc_counts,
            exec_times_without_lsh,
            marker="s",
            markersize=8,
            linewidth=2,
            color="red",
            label="Without LSH",
        )

        # Add value labels on points
        for x, y in zip(doc_counts, exec_times_with_lsh):
            plt.annotate(
                f"{y:.3f}s",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="blue",
            )

        for x, y in zip(doc_counts, exec_times_without_lsh):
            plt.annotate(
                f"{y:.3f}s",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
                color="red",
            )

        plt.title(
            "Scalability Analysis: Execution Time vs Number of Documents\n(With vs Without LSH)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Number of Documents", fontsize=12)
        plt.ylabel("Execution Time (seconds)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # Add description text
        description = (
            "Description: This graph compares execution time with and without LSH for finding similar documents.\n"
            "The blue line (With LSH) only computes similarities for candidate pairs identified by LSH.\n"
            "The red line (Without LSH) computes all pairwise similarities (O(n²) complexity).\n"
            "LSH significantly reduces computation time by filtering candidate pairs before similarity calculation,\n"
            "demonstrating the efficiency benefit of using LSH for large-scale similarity search."
        )
        plt.figtext(
            0.5,
            0.02,
            description,
            ha="center",
            fontsize=9,
            style="italic",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig("execution_time_graph.png", dpi=150, bbox_inches="tight")
        print("✓ Execution time comparison graph saved to: execution_time_graph.png")
        plt.show()

    def plot_jaccard_vs_estimated(self, result):
        """
        Scatter plot of True vs Estimated Similarity.
        This validates the accuracy of MinHash approximation.
        """
        jaccard_dict = {pair: val for pair, val in result["jaccard_results"]}
        estimated_dict = {pair: val for pair, val in result["estimated_results"]}
        common_pairs = set(jaccard_dict.keys()) & set(estimated_dict.keys())

        true_vals = [jaccard_dict[p] for p in common_pairs]
        est_vals = [estimated_dict[p] for p in common_pairs]

        plt.figure(figsize=(8, 9))
        plt.scatter(true_vals, est_vals, color="purple", alpha=0.6)
        plt.plot([0, 1], [0, 1], "--", color="red", label="Perfect Match")
        plt.title(
            "True Jaccard vs Estimated MinHash Similarity",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("True Jaccard Similarity", fontsize=11)
        plt.ylabel("Estimated MinHash Similarity", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add description text
        description = (
            "Description: This graph validates the accuracy of MinHash signatures.\n"
            "Each point represents a document pair. The x-axis shows the true Jaccard\n"
            "similarity (computed from shingle sets), while the y-axis shows the estimated\n"
            "similarity from MinHash signatures. Points close to the red diagonal line\n"
            "indicate accurate estimation. This demonstrates that MinHash provides a good\n"
            "approximation of Jaccard similarity while being much more space-efficient."
        )
        plt.figtext(
            0.5,
            0.02,
            description,
            ha="center",
            fontsize=9,
            style="italic",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.3),
        )

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig("jaccard_vs_estimated_graph.png", dpi=150, bbox_inches="tight")
        print("✓ Jaccard vs Estimated graph saved to: jaccard_vs_estimated_graph.png")
        plt.show()

    def plot_candidates_vs_docs(self, results):
        """
        Plot LSH candidate pairs vs number of documents.
        This shows the efficiency of LSH in finding similar document pairs.
        """
        plt.figure(figsize=(10, 7))
        doc_counts = [r["num_docs"] for r in results]
        candidates = [r["num_candidates"] for r in results]

        bars = plt.bar(
            doc_counts,
            candidates,
            color="orange",
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, candidates):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.title(
            "Number of LSH Candidate Pairs vs Dataset Size",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Number of Documents", fontsize=12)
        plt.ylabel("Candidate Pairs Found", fontsize=12)
        plt.grid(True, alpha=0.3, axis="y")

        # Add description text
        description = (
            "Description: This graph shows how many candidate pairs LSH identifies as potentially\n"
            "similar (above threshold) for different dataset sizes. LSH uses banding technique to\n"
            "efficiently filter document pairs without computing all pairwise similarities. The number\n"
            "of candidates grows with dataset size, but LSH avoids the O(n²) complexity of brute-force\n"
            "comparison by only checking pairs that hash to the same bucket."
        )
        plt.figtext(
            0.5,
            0.02,
            description,
            ha="center",
            fontsize=9,
            style="italic",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="peachpuff", alpha=0.3),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig("candidates_vs_docs_graph.png", dpi=150, bbox_inches="tight")
        print("✓ LSH candidates graph saved to: candidates_vs_docs_graph.png")
        plt.show()


# -------------------------------------------------------
# Run the performance evaluation
# -------------------------------------------------------
if __name__ == "__main__":
    base_path = "Resources/CSV_Data"  # Update path if needed
    similarity_threshold = 0.8  # Similarity threshold: two documents are similar if Jaccard similarity >= threshold

    tester = PerformanceTest(base_path=base_path, k=9, num_perm=1000)
    tester.run(similarity_threshold=similarity_threshold)
