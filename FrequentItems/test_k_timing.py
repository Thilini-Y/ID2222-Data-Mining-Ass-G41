import os
import csv
import matplotlib.pyplot as plt
from frequentItemset import FrequentItemSets


def test_k_timing(support_percentage=0.005, confidence_threshold=0.6):
    """
    Test execution time for finding frequent itemsets for each k value (k=1, k=2, ...)
    """
    base_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_directory, "Resources", "T10I4D100K.dat")
    
    print("=" * 60)
    print("Testing Execution Time for Each k Value")
    print("=" * 60)
    print(f"Dataset: {file_path}")
    print(f"Support threshold: {support_percentage}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 60)
    
    # Initialize the frequent itemset miner
    frequent_items = FrequentItemSets(
        path=file_path,
        support_percentage=support_percentage,
        confidence_threshold=confidence_threshold
    )
    
    # Extract frequent itemsets with k-level timing
    result = frequent_items.extract_frequent_items_with_k_timing()
    
    # Display results
    print("\n" + "=" * 60)
    print("Execution Time Summary by k Value")
    print("=" * 60)
    print(f"{'k':<5} {'Time (s)':<15} {'# Itemsets':<15} {'% of Total':<15}")
    print("-" * 60)
    
    k_timings = result["k_timings"]
    total_time = result["runtime"]
    
    for timing in k_timings:
        k = timing["k"]
        time_k = timing["time"]
        count = timing["count"]
        percentage = (time_k / total_time) * 100 if total_time > 0 else 0
        print(f"{k:<5} {time_k:<15.2f} {count:<15} {percentage:<15.2f}%")
    
    print("-" * 60)
    print(f"{'Total':<5} {total_time:<15.2f} {result['n_itemsets']:<15} {'100.00%':<15}")
    print("=" * 60)
    
    # Save results to CSV
    output_file = os.path.join(base_directory, "k_timing_results.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "Time (s)", "# Itemsets", "% of Total"])
        for timing in k_timings:
            k = timing["k"]
            time_k = timing["time"]
            count = timing["count"]
            percentage = (time_k / total_time) * 100 if total_time > 0 else 0
            writer.writerow([k, f"{time_k:.2f}", count, f"{percentage:.2f}"])
        writer.writerow(["Total", f"{total_time:.2f}", result["n_itemsets"], "100.00"])
    
    print(f"\nResults saved to: {output_file}")
    
    # Create visualization plots
    plot_results(k_timings, total_time, base_directory, support_percentage, confidence_threshold)
    
    return result


def plot_results(k_timings, total_time, base_directory, support_percentage, confidence_threshold):
    """
    Create a graph visualizing execution time for each k value
    """
    # Extract data for plotting
    k_values = [timing["k"] for timing in k_timings]
    times = [timing["time"] for timing in k_timings]
    
    # Create a single figure
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, times, marker='o', linewidth=2.5, markersize=10, color='tab:blue', 
             markerfacecolor='white', markeredgewidth=2, markeredgecolor='tab:blue')
    plt.xlabel('k (Itemset Size)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    plt.title(f'Execution Time vs k Value\n(Support: {support_percentage}, Confidence: {confidence_threshold})', 
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(k_values)
    
    # Add value labels on each point
    for k, time in zip(k_values, times):
        plt.annotate(f'{time:.2f}s', (k, time), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(base_directory, "k_timing_analysis.png")
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()


if __name__ == "__main__":
    # You can modify these parameters as needed
    test_k_timing(support_percentage=0.005, confidence_threshold=0.6)

