import matplotlib.pyplot as plt
import os
import time

from Streams import TriestImproved, TriestBase, ExactCounter, read_dataset


def run_for_sample_size(dataset_path, M):
    # Initialize algorithms
    triest_impr = TriestImproved(M)
    triest_base = TriestBase(M)
    exact = ExactCounter()

    # ===============================
    # TRIEST-IMPR execution time
    # ===============================
    start_impr = time.time()
    for u, v in read_dataset(dataset_path):
        triest_impr.process_edge(u, v)
    impr_time = time.time() - start_impr

    # ===============================
    # TRIEST-BASE execution time
    # ===============================
    triest_base = TriestBase(M)
    start_base = time.time()
    for u, v in read_dataset(dataset_path):
        triest_base.process_edge(u, v)
    base_time = time.time() - start_base

    # ===============================
    # EXACT (for truth) — NOT plotted
    # ===============================
    exact = ExactCounter()
    for u, v in read_dataset(dataset_path):
        exact.process_edge(u, v)

    # Extract values
    est_impr = triest_impr.get_estimation()
    est_base = triest_base.get_estimation()
    true = exact.count

    mape_impr = abs(est_impr - true) / true
    mape_base = abs(est_base - true) / true

    return (
        est_impr, est_base, true,
        mape_impr, mape_base,
        impr_time, base_time
    )


def plot_execution_time_only(sample_sizes, impr_times, base_times):
    """Standalone execution time plot for only IMPR + BASE."""
    plt.figure(figsize=(12, 7))

    # TRIEST-IMPR time
    plt.plot(
        sample_sizes, impr_times,
        marker='o', color="green",
        linewidth=2, markersize=8,
        label="TRIEST-IMPR Time"
    )

    # TRIEST-BASE time
    plt.plot(
        sample_sizes, base_times,
        marker='o', color="blue",
        linewidth=2, markersize=8,
        label="TRIEST-BASE Time"
    )

    plt.xlabel("Sample Size (M)", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title("Execution Time vs Sample Size (M)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=12)

    plt.savefig("execution_time_vs_M.png", dpi=300)
    plt.close()


def plot_results(sample_sizes, est_impr, est_base, true_val,
                 mape_impr, mape_base, impr_times, base_times):

    # ---------------------------------------------------------
    # Plot 1: Triangle Estimates vs M
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, est_impr, marker='o', label="TRIEST-IMPR Estimate")
    plt.plot(sample_sizes, est_base, marker='o', label="TRIEST-BASE Estimate")
    plt.axhline(true_val, color='red', linestyle='--', label="Exact Triangle Count")
    plt.xlabel("Sample Size (M)")
    plt.ylabel("Triangle Count")
    plt.title("Triangle Estimates vs Sample Size (M)")
    plt.grid(True)
    plt.legend()
    plt.savefig("triangle_estimate_vs_M.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: MAPE vs M
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, mape_impr, marker='o', color="purple", label="TRIEST-IMPR MAPE")
    plt.plot(sample_sizes, mape_base, marker='o', color="orange", label="TRIEST-BASE MAPE")
    plt.xlabel("Sample Size (M)")
    plt.ylabel("Error")
    plt.title("Error vs Sample Size (M)")
    plt.grid(True)
    plt.legend()
    plt.savefig("mape_vs_M.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Combined Execution Time vs M (IMPR + BASE only)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, impr_times, marker='o', label="TRIEST-IMPR Time", color="green")
    plt.plot(sample_sizes, base_times, marker='o', label="TRIEST-BASE Time", color="blue")
    plt.xlabel("Sample Size (M)")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig("execution_time_comparison.png", dpi=300)
    plt.close()


def plot_triest_graphs():
    dataset_path = "DataSet/web-Stanford.txt"

    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    sample_sizes = [5000, 10000, 20000, 40000, 60000, 80000, 100000]

    est_impr = []
    est_base = []
    mapes_impr = []
    mapes_base = []
    impr_times = []
    base_times = []

    true_val = None

    print("Running experiments for multiple sample sizes...\n")

    for M in sample_sizes:
        print(f"Processing M = {M} ...")
        (
            e_impr, e_base, true,
            m_impr, m_base,
            t_impr, t_base
        ) = run_for_sample_size(dataset_path, M)

        est_impr.append(e_impr)
        est_base.append(e_base)
        mapes_impr.append(m_impr)
        mapes_base.append(m_base)
        impr_times.append(t_impr)
        base_times.append(t_base)

        if true_val is None:
            true_val = true

    print("\nSaving images...")

    # Standard plots
    plot_results(sample_sizes, est_impr, est_base, true_val,
                 mapes_impr, mapes_base, impr_times, base_times)

    # Standalone IMPR + BASE execution-time plot
    plot_execution_time_only(sample_sizes, impr_times, base_times)

    print("Images saved:")
    print(" - triangle_estimate_vs_M.png")
    print(" - mape_vs_M.png")
    print(" - execution_time_comparison.png")
    print(" - execution_time_vs_M.png  (standalone plot)")


if __name__ == "__main__":
    plot_triest_graphs()
