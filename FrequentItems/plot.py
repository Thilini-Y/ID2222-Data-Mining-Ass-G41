import os
import csv
import matplotlib.pyplot as plt
from frequentItemset import FrequentItemSets


class AprioriRunner:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.results_support = []
        self.results_confidence = []

    def run_support_experiments(self, support_values, confidence=0.6):
        print("\n🔹 Running support threshold experiments...")
        for s in support_values:
            print(f"\n>>> Running Apriori with support={s}")
            miner = FrequentItemSets(path=self.dataset_path,
                                     support_percentage=s,
                                     confidence_threshold=confidence)
            result = miner.extract_frequent_items()
            self.results_support.append(result)

    def run_confidence_experiments(self, confidence_values, support=0.005):
        print("\n🔹 Running confidence threshold experiments...")
        for c in confidence_values:
            print(f"\n>>> Running Apriori with confidence={c}")
            miner = FrequentItemSets(path=self.dataset_path,
                                     support_percentage=support,
                                     confidence_threshold=c)
            result = miner.extract_frequent_items()
            self.results_confidence.append(result)

    def plot_results(self):
        print("\n📊 Drawing figures...")

        # Extract values for plotting
        supports = [r["support"] for r in self.results_support]
        durations = [r["runtime"] for r in self.results_support]
        n_freq_items = [r["n_itemsets"] for r in self.results_support]

        confidences = [r["confidence"] for r in self.results_confidence]
        durations_conf = [r["runtime"] for r in self.results_confidence]
        n_rules = [r["n_rules"] for r in self.results_confidence]

        # ----- 1. Runtime vs Support -----
        plt.figure(figsize=(6, 4))
        plt.plot(supports, durations, marker="o", color="tab:blue")
        plt.xlabel("Support threshold")
        plt.ylabel("Execution time (s)")
        plt.title("Execution Time vs Support Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("runtime_vs_support.png", dpi=200)
        plt.show()

        # ----- 2. Frequent Itemsets vs Support -----
        plt.figure(figsize=(6, 4))
        plt.plot(supports, n_freq_items, marker="o", color="tab:orange")
        plt.xlabel("Support threshold")
        plt.ylabel("Number of frequent itemsets")
        plt.title("Frequent Itemsets vs Support Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("freq_itemsets_vs_support.png", dpi=200)
        plt.show()

        # ----- 3. Runtime vs Confidence -----
        plt.figure(figsize=(6, 4))
        plt.plot(confidences, durations_conf, marker="o", color="tab:green")
        plt.xlabel("Confidence threshold")
        plt.ylabel("Execution time (s)")
        plt.title("Execution Time vs Confidence Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("runtime_vs_confidence.png", dpi=200)
        plt.show()

        # ----- 4. Number of Rules vs Confidence -----
        plt.figure(figsize=(6, 4))
        plt.plot(confidences, n_rules, marker="o", color="tab:red")
        plt.xlabel("Confidence threshold")
        plt.ylabel("Number of rules")
        plt.title("Number of Rules vs Confidence Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rules_vs_confidence.png", dpi=200)
        plt.show()

        print("✅ All plots saved as PNG images.")

    def save_results_to_csv(self):
        with open("experiment_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "Threshold", "Runtime(s)", "#Itemsets", "#Rules"])
            for r in self.results_support:
                writer.writerow(["Support", r["support"], r["runtime"],
                                 r["n_itemsets"], r["n_rules"]])
            for r in self.results_confidence:
                writer.writerow(["Confidence", r["confidence"], r["runtime"],
                                 r["n_itemsets"], r["n_rules"]])
        print("📁 Results saved to experiment_results.csv")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "resources", "T10I4D100K.dat")

    # Initialize runner
    runner = AprioriRunner(dataset_path=file_path)

    # Define thresholds for experiments
    support_values = [0.02, 0.01, 0.005]
    confidence_values = [0.4, 0.5, 0.6, 0.7, 0.8]

    # Run experiments
    runner.run_support_experiments(support_values)
    runner.run_confidence_experiments(confidence_values)

    # Generate plots and save results
    runner.plot_results()
    runner.save_results_to_csv()

    print("\n🎯 Experiment complete. You can include plots and CSV data in your report.")
