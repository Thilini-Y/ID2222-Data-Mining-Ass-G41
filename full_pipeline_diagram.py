#!/usr/bin/env python3
import os
import time
from typing import List
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

from compareSignature import CompareSignatures
from lsh import LSH
from minHashing import MinHashing
from shingling import Shingling


class FullPipelineGraphs:
    """
    Measure execution time of the full pipeline: Shingling → MinHash → LSH → Similarity.
    Shows how the complete flow scales with varying document counts.
    """

    def __init__(
        self,
        k: int = 9,
        num_perm: int = 1000,
        bands: int = 50,
        rows_per_band: int = 20,
        app_name: str = "FullPipelineExecutionTime",
    ):
        self.k = k
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = rows_per_band

        self.spark = (
            SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
        )
        self.spark_context = self.spark.sparkContext

    def load_documents(self, path: str, num_files: int):
        """
        Read up to num_files files from path on the driver, then parallelize the resulting list.
        This avoids opening local files inside a Spark worker process.
        Returns an RDD of (filename, content).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        files = [
            f
            for f in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")
        ]

        if not files:
            raise ValueError(f"No files found in folder: {path}")

        selected_files = files[:num_files]
        print(f"Folder: {path}")
        print(f"Files selected ({len(selected_files)}): {selected_files}")

        docs = []
        for fname in selected_files:
            fpath = os.path.join(path, fname)
            try:
                with open(fpath, "r", errors="ignore") as fh:
                    text = fh.read()
            except Exception as e:
                print(f"  ! Warning: Could not read {fpath}: {e}")
                text = ""
            docs.append((fname, text))

        return self.spark_context.parallelize(docs)

    def run(
        self,
        folder_path: str,
        num_files_list: List[int],
        similarity_threshold: float = None,
    ):
        """
        Run full pipeline execution time measurement for different document counts.
        similarity_threshold: if set, stores only similarities >= threshold (not required).
        """
        execution_times = []
        detailed_stage_times = {}
        x_values = sorted(set(num_files_list))

        # Warm-up run to initialize Spark (excluded from measurements)
        if x_values:
            print("\n========== Warm-up run (Spark initialization) ==========")
            first_num_files = x_values[0]
            documents_rdd = self.load_documents(folder_path, first_num_files)

            # Create local variables to avoid serialization issues
            k = self.k
            shingled_rdd = documents_rdd.map(
                lambda x: (x[0], Shingling(k).create_shingles(x[1]))
            )
            shingled_rdd.persist()
            shingled_rdd.map(lambda x: x[0]).collect()  # Trigger computation
            shingled_rdd.unpersist()
            print("  ✓ Spark warm-up complete\n")

        for num_files in x_values:
            print(
                f"\n========== Running full pipeline for {num_files} files =========="
            )

            overall_start = time.time()

            # Step 1: Load documents (driver-side read + parallelize)
            t0 = time.time()
            documents_rdd = self.load_documents(folder_path, num_files)
            t1 = time.time()
            load_time = t1 - t0
            print(f"  → Loaded and parallelized {num_files} files in {load_time:.4f} s")

            # Step 2: Create shingles for each document
            # Use local variables to avoid serialization issues
            t0 = time.time()
            k = self.k
            shingled_rdd = documents_rdd.map(
                lambda x: (x[0], Shingling(k).create_shingles(x[1]))
            )
            # persist if re-used
            shingled_rdd = shingled_rdd.persist()
            # small collect to check count (avoid collecting contents)
            docs_meta = shingled_rdd.map(lambda x: x[0]).collect()
            shingle_time = time.time() - t0
            print(
                f"  → Created shingles for {len(docs_meta)} documents in {shingle_time:.4f} s"
            )

            # Step 3: Compute MinHash signatures
            t0 = time.time()
            num_perm = self.num_perm

            # Create a function that will be serialized properly
            def compute_minhash(shingles):
                minhasher = MinHashing(num_perm=num_perm)
                return minhasher.compute_for_doc(
                    shingles,
                    getattr(minhasher, "hash_params", None),
                    getattr(minhasher, "prime", None),
                )

            signatures_rdd = shingled_rdd.mapValues(compute_minhash)
            # Bring signatures to driver as a dict (required by LSH implementation below)
            doc_signatures = dict(signatures_rdd.collect())
            minhash_time = time.time() - t0
            print(
                f"  → Computed MinHash signatures for {len(doc_signatures)} docs in {minhash_time:.4f} s"
            )

            # Step 4: Use LSH to find candidate pairs
            t0 = time.time()
            lsh = LSH(bands=self.bands, rows_per_band=self.rows_per_band)
            candidates = lsh.find_candidates(doc_signatures)
            lsh_time = time.time() - t0
            print(
                f"  → Found {len(candidates)} candidate pairs using LSH in {lsh_time:.4f} s"
            )

            # Step 5: Compute similarities for candidate pairs
            t0 = time.time()
            similarities = []
            for f1, f2 in candidates:
                sig1, sig2 = doc_signatures[f1], doc_signatures[f2]
                sim = CompareSignatures.similarity(sig1, sig2)
                if similarity_threshold is None or sim >= similarity_threshold:
                    similarities.append(((f1, f2), sim))
            similarity_time = time.time() - t0
            print(
                f"  → Computed similarities for candidate pairs in {similarity_time:.4f} s"
            )

            overall_end = time.time()
            execution_time = overall_end - overall_start
            execution_times.append(execution_time)
            detailed_stage_times[num_files] = {
                "load": load_time,
                "shingle": shingle_time,
                "minhash": minhash_time,
                "lsh": lsh_time,
                "similarity": similarity_time,
                "total": execution_time,
                "num_candidates": len(candidates),
                "num_similar_pairs": len(similarities),
            }

            print(f"  ✓ Full pipeline execution time: {execution_time:.4f} seconds")
            print(
                f"  ✓ Processed {len(candidates)} candidate pairs, {len(similarities)} passed threshold"
            )

            # unpersist shingled_rdd before next iteration
            try:
                shingled_rdd.unpersist()
            except Exception:
                pass

        # Stop Spark and plot
        self.spark.stop()
        self.plot_results(x_values, execution_times, detailed_stage_times)

    def plot_results(self, num_files_list, execution_times, stage_times=None):
        """Plot execution time of the full pipeline vs number of documents."""
        plt.figure(figsize=(12, 6.5))

        # Execution time line
        plt.plot(
            num_files_list,
            execution_times,
            "o-g",
            linewidth=2,
            markersize=8,
            label="Full Pipeline Execution Time",
        )

        # Add value labels on points
        for x, y in zip(num_files_list, execution_times):
            plt.text(
                x, y, f"{y:.3f}s", color="green", fontsize=9, ha="left", va="bottom"
            )

        plt.title(
            "Full Pipeline Execution Time: Shingling → MinHash → LSH → Similarity\n(Scalability Analysis)",
            fontsize=15,
            fontweight="bold",
        )
        plt.xlabel("Number of Documents", fontsize=12)
        plt.ylabel("Execution Time (seconds)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)

        desc = (
            "Description: This graph shows the end-to-end execution time of the complete pipeline:\n"
            "1. Shingling: Convert documents to k-shingles\n"
            "2. MinHash: Compute signature vectors for each document\n"
            "3. LSH: Use Locality-Sensitive Hashing to find candidate pairs\n"
            "4. Similarity: Compute MinHash similarity for candidate pairs\n"
            "The pipeline efficiently scales by using LSH to filter candidate pairs before similarity computation."
        )
        plt.text(
            0.5,
            -0.32,
            desc,
            ha="center",
            va="top",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor="lightgreen", alpha=0.6, boxstyle="round,pad=0.5"),
        )

        plt.tight_layout()
        out_fname = "./images/full_pipeline_execution_time.png"
        plt.savefig(out_fname, dpi=150, bbox_inches="tight")
        print(f"✓ Full pipeline execution time graph saved to: {out_fname}")
        plt.show()

        # Print stage breakdown if requested
        if stage_times:
            print("\nStage breakdown (seconds):")
            for n in sorted(stage_times.keys()):
                st = stage_times[n]
                print(
                    f"  {n:>5} files → load: {st['load']:.3f}, shingle: {st['shingle']:.3f}, "
                    f"minhash: {st['minhash']:.3f}, lsh: {st['lsh']:.3f}, similarity: {st['similarity']:.3f}, total: {st['total']:.3f}"
                )


if __name__ == "__main__":
    base_path = "Resources/Total_data"
    # tune these numbers as needed; ensure your MinHashing produces num_perm-length signatures
    app = FullPipelineGraphs(k=9, num_perm=1000, bands=50, rows_per_band=20)
    # choose which document counts to test; can be any integers <= number of files present
    app.run(base_path, num_files_list=[50, 100, 200, 400, 600, 800, 1000])
