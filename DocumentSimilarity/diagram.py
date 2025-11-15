import itertools
import os
import time
from typing import List
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from compareSets import CompareSets
from compareSignature import CompareSignatures
from lsh import LSH
from minHashing import MinHashing
from shingling import Shingling


class Graphs:
    def __init__(self, k: int = 9, num_perm: int = 1000):
        self.k = k
        self.shingler = Shingling(k)
        self.num_perm = num_perm
        self.minhasher = MinHashing(num_perm=num_perm)
        self.spark = (
            SparkSession.builder.appName("GraphsExecutionComparison")
            .master("local[*]")
            .getOrCreate()
        )
        self.spark_context = self.spark.sparkContext

    def load_documents(self, path: str, num_files: int):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        files = [
            f
            for f in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")
        ]

        if not files:
            raise ValueError(f"No files found in folder: {path}")

        selected_files = [(f, os.path.join(path, f)) for f in files[:num_files]]

        print(f"Folder: {path}")
        print(
            f"Files selected ({len(selected_files)}): {[f for f, _ in selected_files]}"
        )

        # read files in parallel using Spark
        files_rdd = self.spark_context.parallelize(selected_files)
        read_files_rdd = files_rdd.map(
            lambda x: (x[0], open(x[1], "r", errors="ignore").read())
        )
        return read_files_rdd

    def run(self, folder_path: str, num_files_list: List[int]):
        jaccard_times = []
        minhash_times = []
        last_similarity_pairs = []
        last_num_files = None

        for num_files in sorted(set(num_files_list)):
            print(f"\n========== Running for {num_files} files ==========")
            documents_rdd = self.load_documents(folder_path, num_files)
            shingler = self.shingler

            # create shingles for each document
            shingled_rdd = documents_rdd.map(
                lambda x: (x[0], shingler.create_shingles(x[1]))
            )
            documents = shingled_rdd.collect()
            print(f"Total documents processed: {len(documents)}")

            # ---- JACCARD TIMING ----
            print("\n********** Jaccard Similarity **********")
            start_jaccard = time.time()
            for (file1, s1), (file2, s2) in itertools.combinations(documents, 2):
                CompareSets.jaccard_similarity(s1, s2)
            end_jaccard = time.time()
            jaccard_time = end_jaccard - start_jaccard
            jaccard_times.append(jaccard_time)
            print(f"Jaccard execution time: {jaccard_time:.4f} seconds")

            # ---- MINHASH TIMING ----
            print("\n********** Computing Similarity using MinHash **********")
            # 🩵 Fix: avoid capturing self inside Spark lambdas
            minhasher = self.minhasher
            start_minhash = time.time()

            signatures_rdd = shingled_rdd.mapValues(
                lambda shingles: minhasher.compute_for_doc(
                    shingles, minhasher.hash_params, minhasher.prime
                )
            )

            doc_signatures = dict(signatures_rdd.collect())
            for (file1, sig1), (file2, sig2) in itertools.combinations(
                doc_signatures.items(), 2
            ):
                CompareSignatures.similarity(sig1, sig2)

            end_minhash = time.time()
            minhash_time = end_minhash - start_minhash
            minhash_times.append(minhash_time)
            print(f"MinHash execution time: {minhash_time:.4f} seconds")

            pairwise_pairs = []
            for (file1, shingles1), (file2, shingles2) in itertools.combinations(
                documents, 2
            ):
                jaccard_val = CompareSets.jaccard_similarity(shingles1, shingles2)
                minhash_val = CompareSignatures.similarity(
                    doc_signatures[file1], doc_signatures[file2]
                )
                pairwise_pairs.append((jaccard_val, minhash_val))

            if pairwise_pairs:
                last_similarity_pairs = pairwise_pairs
                last_num_files = num_files

        self.spark.stop()
        self.plot_results(num_files_list, jaccard_times, minhash_times)
        if last_similarity_pairs:
            self.plot_similarity_comparison(last_similarity_pairs, last_num_files)

    def plot_results(self, num_files_list, jaccard_times, minhash_times):
        plt.figure(figsize=(12, 6.5))

        # Jaccard line
        plt.plot(
            num_files_list,
            jaccard_times,
            "o-b",
            linewidth=2,
            markersize=8,
            label="Jaccard Similarity",
        )
        for x, y in zip(num_files_list, jaccard_times):
            plt.text(
                x, y, f"{y:.3f}s", color="blue", fontsize=9, ha="left", va="bottom"
            )

        # MinHash line
        plt.plot(
            num_files_list,
            minhash_times,
            "s-r",
            linewidth=2,
            markersize=8,
            label="MinHash Similarity",
        )
        for x, y in zip(num_files_list, minhash_times):
            plt.text(x, y, f"{y:.3f}s", color="red", fontsize=9, ha="left", va="bottom")

        plt.title(
            "Execution Time Comparison: Jaccard vs MinHash Similarity\n(All Pairwise Comparisons)",
            fontsize=15,
            fontweight="bold",
        )
        plt.xlabel("Number of Documents", fontsize=12)
        plt.ylabel("Execution Time (seconds)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        desc = (
            "Description: This graph compares end-to-end execution time for computing similarity between all document pairs.\n"
            "Jaccard (blue) computes directly on shingle sets using Python set operations.\n"
            "MinHash (red) computes signatures first (num_perm × num_shingles ops per doc), enabling LSH.\n"
            "For small datasets, Jaccard is faster; for larger or approximate tasks, MinHash scales better."
        )
        plt.text(
            0.5,
            -0.28,
            desc,
            ha="center",
            va="top",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor="beige", alpha=0.6, boxstyle="round,pad=0.5"),
        )

        plt.tight_layout()
        out_fname = "./images/JaccardVsMinHashing.png"
        plt.savefig(out_fname, dpi=150, bbox_inches="tight")
        print(f"Jaccard vs MinHashing execution time graph saved to: {out_fname}")
        plt.show()

    def plot_similarity_comparison(self, pairwise_pairs, num_files):
        if not pairwise_pairs:
            print("No pairwise similarities available to plot.")
            return

        jaccard_values = [pair[0] for pair in pairwise_pairs]
        minhash_values = [pair[1] for pair in pairwise_pairs]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            jaccard_values,
            minhash_values,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.4,
            label="Pairs",
        )
        plt.plot([0, 1], [0, 1], "k--", label="Ideal agreement")

        title_suffix = f" ({num_files} documents)" if num_files else ""
        plt.title(
            "True Jaccard vs Estimated MinHash Similarity" + title_suffix,
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("True Jaccard Similarity", fontsize=12)
        plt.ylabel("Estimated MinHash Similarity", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        os.makedirs("images", exist_ok=True)
        suffix = f"_{num_files}" if num_files else ""
        out_fname = f"./images/Jaccard_vs_MinHashSimilarity{suffix}.png"
        plt.savefig(out_fname, dpi=150, bbox_inches="tight")
        print(
            "True Jaccard vs estimated MinHash similarity scatter plot saved to: "
            f"{out_fname}"
        )
        plt.show()


if __name__ == "__main__":
    base_path = "Resources/Total_data"
    app = Graphs(k=9, num_perm=1000)
    app.run(base_path, num_files_list=[50, 100, 200, 400, 600, 800, 1000])
