import os
import time
from itertools import combinations
from pyspark.sql import SparkSession


class FrequentItemSets:
    def __init__(self, path, support_percentage=0.005, confidence_threshold=0.6 ):
        self.path = path
        self.support_percentage = support_percentage
        self.confidence_threshold = confidence_threshold
        self.spark = (
            SparkSession.builder
            .appName("Apriori-Frequent-Itemsets-Optimized")
            .master("local[*]")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.default.parallelism", "8")
            .getOrCreate()
        )

    @staticmethod
    def _get_candidates(frequent_item_sets, k):
        # pruned candidate itemsets
        items = set(i for t in frequent_item_sets for i in t)
        candidates = [tuple(sorted(c)) for c in combinations(items, k)]

        # keep only candidates whose k-1 subsets are frequent
        frequent_sets = [set(f) for f in frequent_item_sets]
        pruned = []
        for c in candidates:
            subsets = combinations(c, k - 1)
            if all(set(subset) in frequent_sets for subset in subsets):
                pruned.append(c)
        return pruned

    def _generate_association_rules(self, frequent_items, support_threshold):
        rules = []
        support_itemsets = frequent_items

        for itemset, support_Ij in support_itemsets.items():
            if len(itemset) < 2:
                continue

            for r in range(1, len(itemset)):
                for combination_x in combinations(itemset, r):
                    i = tuple(sorted(combination_x))
                    j = tuple(sorted(item for item in itemset if item not in i))

                    support_I = support_itemsets.get(i)
                    if not support_I:
                        continue

                    confidence = support_Ij / support_I
                    if confidence >= self.confidence_threshold:
                        rules.append((i,j, support_Ij, confidence))

        rules.sort(key=lambda r: (r[2], r[3], len(r[0]) + len(r[1])), reverse=True)
        return rules

    def extract_frequent_items(self):
        start_time = time.time()

        # Load dataset
        data_frame = self.spark.read.text(self.path)
        baskets = (
            data_frame.rdd
            .map(lambda row: set(row.value.strip().split()))
            .filter(lambda b: len(b) > 0)
            .repartition(8)
            .cache()
        )

        total_baskets = baskets.count()
        support_threshold = int(self.support_percentage * total_baskets)

        print(f"Total baskets: {total_baskets}")
        print(f"Support threshold: {support_threshold}")

        # C1, L1
        print(f"\n--- Processing k = 1 ---")
        item_count = baskets.flatMap(lambda b: [(item, 1) for item in b]).reduceByKey(lambda x, y: x + y)
        frequent_items_l = item_count.filter(lambda i: i[1] >= support_threshold).cache()

        l = [tuple([item[0]]) for item in frequent_items_l.collect()]
        l_all = [(tuple([item[0]]), item[1]) for item in frequent_items_l.collect()]

        print(f"Found {len(l)} frequent itemsets for k = 1")

        k = 2
        while l:
            print(f"\n--- Processing k = {k} ---")

            candidates = self._get_candidates(l, k)
            if not candidates:
                break

            # broadcast candidates to executors
            broadcast_candidates = self.spark.sparkContext.broadcast(candidates)

            candidate_pairs = baskets.flatMap(
                lambda basket: [
                    (c, 1) for c in broadcast_candidates.value if set(c).issubset(basket)
                ]
            ).reduceByKey(lambda a, b: a + b)

            candidate_counts = (
                candidate_pairs
                .filter(lambda x: x[1] >= support_threshold)
                .cache()
            )

            count_l = candidate_counts.count()
            if count_l == 0:
                print(f"No frequent itemsets found for size {k}.")
                break

            l = [item[0] for item in candidate_counts.collect()]
            l_all.extend(candidate_counts.collect())

            print(f"Found {count_l} frequent itemsets for k = {k}")
            k += 1

        print("\n***** Frequent Itemsets *****")
        for itemset, count in l_all:
            print(f"{itemset}: {count}")

        freq_itemset = {tuple(sorted(itemset)): count for itemset, count in l_all}
        rules = self._generate_association_rules(freq_itemset, support_threshold)

        print("\n***********Association rules rules (X -> Y)   [support, confidence]:***************")
        for i, (x, y, support, confidence) in enumerate(rules, 1):
            print(f"{i:>2}. {x} -> {y}   [support={support}, confidence={confidence:.3f}]")

        elapsed = time.time() - start_time
        print(f"\nExecution time: {elapsed:.2f} seconds")

        self.spark.stop()

        return {
            "support": self.support_percentage,
            "confidence": self.confidence_threshold,
            "runtime": elapsed,
            "n_itemsets": len(freq_itemset),
            "n_rules": len(rules)
        }


if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_directory, "resources", "T10I4D100K.dat")

    frequent_items = FrequentItemSets(path=file_path, support_percentage=0.005, confidence_threshold = 0.6)
    frequent_items.extract_frequent_items()
