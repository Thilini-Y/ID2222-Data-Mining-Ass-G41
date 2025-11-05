# Scalability test: execution time vs dataset size
# Tests with 5, 10, and more documents to measure performance
import time
import os
from documentSimilarity import DocumentSimilarity


def scalability_test(base_path, similarity_threshold=0.8):
    """
    Test scalability by measuring execution time for different dataset sizes.
    Uses similarity threshold s (default 0.8) to find similar documents.
    """
    print("=" * 70)
    print("SCALABILITY TEST: Execution Time vs Dataset Size")
    print("=" * 70)
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Dataset path: {base_path}")
    print()

    # Check available files
    if not os.path.exists(base_path):
        print(f"Error: Dataset folder not found: {base_path}")
        return

    all_files = [
        f
        for f in os.listdir(base_path)
        if os.path.isfile(os.path.join(base_path, f)) and not f.startswith(".")
    ]

    if not all_files:
        print(f"Error: No files found in {base_path}")
        return

    print(f"Available files in dataset: {len(all_files)}")
    print()

    # Test with different dataset sizes: 5, 10, and progressive sizes up to available
    test_sizes = [5, 10]

    # Add more sizes if we have enough files
    if len(all_files) >= 12:
        test_sizes.append(12)
    if len(all_files) >= 20:
        test_sizes.append(20)
    if len(all_files) >= 50:
        test_sizes.append(50)
    if len(all_files) >= 100:
        test_sizes.append(100)
    if len(all_files) >= 500:
        test_sizes.append(500)
    if len(all_files) >= 1000:
        test_sizes.append(1000)

    # Always test with all available files if > 10
    if len(all_files) > 10:
        test_sizes.append(len(all_files))

    results = []

    for num_files in test_sizes:
        if num_files > len(all_files):
            continue

        print(f"\n{'='*70}")
        print(f"Testing with {num_files} documents")
        print(f"{'='*70}")

        # Create app instance
        app = DocumentSimilarity(k=9)

        # Measure execution time
        start_time = time.time()

        try:
            # Run the similarity analysis
            app.run(base_path, num_files=num_files)

            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate number of pairs (n choose 2)
            num_pairs = num_files * (num_files - 1) // 2

            results.append(
                {
                    "num_files": num_files,
                    "num_pairs": num_pairs,
                    "execution_time": execution_time,
                }
            )

            print(f"\nExecution time: {execution_time:.4f} seconds")
            print(f"Number of document pairs: {num_pairs}")
            print(f"Time per pair: {execution_time/num_pairs*1000:.4f} ms")

        except Exception as e:
            print(f"Error testing with {num_files} files: {e}")
            app.spark.stop()
            continue

        app.spark.stop()

    # Summary table
    print("\n" + "=" * 70)
    print("SCALABILITY SUMMARY")
    print("=" * 70)
    print(f"{'Files':<10} {'Pairs':<10} {'Time (s)':<15} {'Time/Pair (ms)':<15}")
    print("-" * 70)

    for r in results:
        time_per_pair = r["execution_time"] / r["num_pairs"] * 1000
        print(
            f"{r['num_files']:<10} {r['num_pairs']:<10} {r['execution_time']:<15.4f} {time_per_pair:<15.4f}"
        )

    # Analyze scaling
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("SCALING ANALYSIS")
        print("=" * 70)

        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            files_ratio = curr["num_files"] / prev["num_files"]
            pairs_ratio = curr["num_pairs"] / prev["num_pairs"]
            time_ratio = curr["execution_time"] / prev["execution_time"]

            print(f"\nFrom {prev['num_files']} to {curr['num_files']} files:")
            print(f"  Files increase: {files_ratio:.2f}x")
            print(f"  Pairs increase: {pairs_ratio:.2f}x")
            print(f"  Time increase: {time_ratio:.2f}x")

            if time_ratio < pairs_ratio:
                print(f"  ✓ Better than linear scaling (time grows slower than pairs)")
            elif time_ratio == pairs_ratio:
                print(f"  → Linear scaling (time grows proportionally with pairs)")
            else:
                print(f"  ⚠ Worse than linear scaling (time grows faster than pairs)")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    base_path = "Resources/Dataset"
    similarity_threshold = 0.8  # Threshold s as specified in requirements

    scalability_test(base_path, similarity_threshold)
