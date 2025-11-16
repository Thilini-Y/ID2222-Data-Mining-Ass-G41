import os
import time
from itertools import combinations
from pyspark.sql import SparkSession


class FrequentItemSets:
    def __init__(self, path, support_percentage=0.005, confidence_threshold=0.6):
        self.path = path
        self.support_percentage = support_percentage
        self.confidence_threshold = confidence_threshold
        self.spark = (
            SparkSession.builder.appName("Apriori-Frequent-Itemsets-Optimized")
            .master("local[*]")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.default.parallelism", "8")
            # OPTIMIZATIONS
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            .config("spark.rdd.compress", "true")
            .config("spark.shuffle.compress", "true")
            .getOrCreate()
        )

    @staticmethod
    def _get_candidates(frequent_item_sets, k):
        """OPTIMIZED: Use frozenset for O(1) lookup instead of O(n*m)"""
        # Convert to frozenset once for fast lookup
        frequent_sets = {frozenset(f) for f in frequent_item_sets}

        # Get all unique items
        items = set(i for t in frequent_item_sets for i in t)

        # Generate all k-combinations
        candidates = [tuple(sorted(c)) for c in combinations(items, k)]

        # Prune candidates whose k-1 subsets are not all frequent
        pruned = []
        for c in candidates:
            # Check if all k-1 subsets are frequent - O(k) with O(1) lookup
            if all(
                frozenset(subset) in frequent_sets for subset in combinations(c, k - 1)
            ):
                pruned.append(c)

        return pruned

    def _generate_association_rules(self, frequent_items, support_threshold):
        """OPTIMIZED: Parallelize rule generation with RDD"""
        # Broadcast data for workers
        support_bc = self.spark.sparkContext.broadcast(frequent_items)
        confidence_threshold_bc = self.spark.sparkContext.broadcast(
            self.confidence_threshold
        )

        def generate_rules_for_itemset(item):
            itemset, support_Ij = item
            if len(itemset) < 2:
                return []

            rules = []
            support_map = support_bc.value

            for r in range(1, len(itemset)):
                for combination_x in combinations(itemset, r):
                    i = tuple(sorted(combination_x))
                    j = tuple(sorted(item for item in itemset if item not in i))

                    support_I = support_map.get(i)
                    if not support_I:
                        continue

                    confidence = support_Ij / support_I
                    if confidence >= confidence_threshold_bc.value:
                        rules.append((i, j, support_Ij, confidence))

            return rules

        # Parallelize rule generation
        itemsets_rdd = self.spark.sparkContext.parallelize(
            list(frequent_items.items()), numSlices=8
        )
        rules = itemsets_rdd.flatMap(generate_rules_for_itemset).collect()

        rules.sort(key=lambda r: (r[2], r[3], len(r[0]) + len(r[1])), reverse=True)
        return rules

    def extract_frequent_items(self):
        start_time = time.time()

        # Load dataset
        data_frame = self.spark.read.text(self.path)
        baskets = (
            data_frame.rdd.map(lambda row: frozenset(row.value.strip().split()))
            .filter(lambda b: len(b) > 0)
            .repartition(8)
            .cache()
        )

        total_baskets = baskets.count()
        support_threshold = int(self.support_percentage * total_baskets)

        print(f"Total baskets: {total_baskets}")
        print(f"Support threshold: {support_threshold}")

        # === PASS 1: Find L1 and filter transactions ===
        print(f"\n--- Processing k = 1 ---")
        item_count = baskets.flatMap(lambda b: [(item, 1) for item in b]).reduceByKey(
            lambda x, y: x + y
        )
        frequent_items_l = item_count.filter(
            lambda i: i[1] >= support_threshold
        ).cache()

        l = [tuple([item[0]]) for item in frequent_items_l.collect()]
        l_all = [(tuple([item[0]]), item[1]) for item in frequent_items_l.collect()]

        print(f"Found {len(l)} frequent itemsets for k = 1")

        # OPTIMIZATION: Filter baskets to only contain frequent items
        frequent_items_set = set(item[0] for item in l)
        frequent_items_bc = self.spark.sparkContext.broadcast(frequent_items_set)

        baskets_filtered = (
            baskets.map(
                lambda b: frozenset(
                    item for item in b if item in frequent_items_bc.value
                )
            )
            .filter(lambda b: len(b) > 1)
            .cache()
        )

        # Force evaluation and unpersist old RDD
        baskets_filtered.count()
        baskets.unpersist()
        baskets = baskets_filtered

        print(f"Filtered transactions to contain only frequent items")

        # === PASS 2+: Find L2, L3, ... ===
        k = 2
        while l:
            print(f"\n--- Processing k = {k} ---")

            candidates = self._get_candidates(l, k)
            if not candidates:
                break

            print(f"Generated {len(candidates)} candidates")

            # OPTIMIZATION: Convert candidates to frozenset for faster subset checking
            candidates_as_sets = [frozenset(c) for c in candidates]
            broadcast_candidates = self.spark.sparkContext.broadcast(candidates_as_sets)

            # Count candidates
            candidate_pairs = baskets.flatMap(
                lambda basket: [
                    (tuple(sorted(c)), 1)
                    for c in broadcast_candidates.value
                    if c.issubset(basket)  # Much faster with frozenset
                ]
            ).reduceByKey(lambda a, b: a + b)

            candidate_counts = candidate_pairs.filter(
                lambda x: x[1] >= support_threshold
            ).cache()

            count_l = candidate_counts.count()
            if count_l == 0:
                print(f"No frequent itemsets found for size {k}.")
                break

            l = [item[0] for item in candidate_counts.collect()]
            l_all.extend(candidate_counts.collect())

            print(f"Found {count_l} frequent itemsets for k = {k}")

            # Clean up broadcast variable
            broadcast_candidates.unpersist()

            k += 1

        print("\n***** Frequent Itemsets *****")
        for itemset, count in l_all:
            print(f"{itemset}: {count}")

        freq_itemset = {tuple(sorted(itemset)): count for itemset, count in l_all}

        print("\n--- Generating Association Rules ---")
        rules = self._generate_association_rules(freq_itemset, support_threshold)

        print(
            "\n***********Association rules (X -> Y)   [support, confidence]:***************"
        )
        for i, (x, y, support, confidence) in enumerate(rules, 1):
            print(
                f"{i:>2}. {x} -> {y}   [support={support}, confidence={confidence:.3f}]"
            )

        elapsed = time.time() - start_time
        print(f"\nExecution time: {elapsed:.2f} seconds")

        # Cleanup
        baskets.unpersist()
        self.spark.stop()

        return {
            "support": self.support_percentage,
            "confidence": self.confidence_threshold,
            "runtime": elapsed,
            "n_itemsets": len(freq_itemset),
            "n_rules": len(rules),
        }

    # This for collecting k-level timing information for performance analysis
    def extract_frequent_items_with_k_timing(self):
        """Extract frequent itemsets and return timing information for each k value"""
        start_time = time.time()
        k_timings = []

        # Load dataset
        data_frame = self.spark.read.text(self.path)
        baskets = (
            data_frame.rdd.map(lambda row: frozenset(row.value.strip().split()))
            .filter(lambda b: len(b) > 0)
            .repartition(8)
            .cache()
        )

        total_baskets = baskets.count()
        support_threshold = int(self.support_percentage * total_baskets)

        print(f"Total baskets: {total_baskets}")
        print(f"Support threshold: {support_threshold}")

        # === PASS 1: Find L1 and filter transactions ===
        print(f"\n--- Processing k = 1 ---")
        k1_start = time.time()

        item_count = baskets.flatMap(lambda b: [(item, 1) for item in b]).reduceByKey(
            lambda x, y: x + y
        )
        frequent_items_l = item_count.filter(
            lambda i: i[1] >= support_threshold
        ).cache()

        l = [tuple([item[0]]) for item in frequent_items_l.collect()]
        l_all = [(tuple([item[0]]), item[1]) for item in frequent_items_l.collect()]

        k1_time = time.time() - k1_start
        k_timings.append({"k": 1, "time": k1_time, "count": len(l)})
        print(f"Found {len(l)} frequent itemsets for k = 1 (Time: {k1_time:.2f}s)")

        # OPTIMIZATION: Filter baskets to only contain frequent items
        frequent_items_set = set(item[0] for item in l)
        frequent_items_bc = self.spark.sparkContext.broadcast(frequent_items_set)

        baskets_filtered = (
            baskets.map(
                lambda b: frozenset(
                    item for item in b if item in frequent_items_bc.value
                )
            )
            .filter(lambda b: len(b) > 1)
            .cache()
        )

        # Force evaluation and unpersist old RDD
        baskets_filtered.count()
        baskets.unpersist()
        baskets = baskets_filtered

        print(f"Filtered transactions to contain only frequent items")

        # === PASS 2+: Find L2, L3, ... ===
        k = 2
        while l:
            print(f"\n--- Processing k = {k} ---")
            k_start = time.time()

            candidates = self._get_candidates(l, k)
            if not candidates:
                break

            print(f"Generated {len(candidates)} candidates")

            # OPTIMIZATION: Convert candidates to frozenset for faster subset checking
            candidates_as_sets = [frozenset(c) for c in candidates]
            broadcast_candidates = self.spark.sparkContext.broadcast(candidates_as_sets)

            # Count candidates
            candidate_pairs = baskets.flatMap(
                lambda basket: [
                    (tuple(sorted(c)), 1)
                    for c in broadcast_candidates.value
                    if c.issubset(basket)  # Much faster with frozenset
                ]
            ).reduceByKey(lambda a, b: a + b)

            candidate_counts = candidate_pairs.filter(
                lambda x: x[1] >= support_threshold
            ).cache()

            count_l = candidate_counts.count()
            if count_l == 0:
                print(f"No frequent itemsets found for size {k}.")
                break

            l = [item[0] for item in candidate_counts.collect()]
            l_all.extend(candidate_counts.collect())

            k_time = time.time() - k_start
            k_timings.append({"k": k, "time": k_time, "count": count_l})
            print(
                f"Found {count_l} frequent itemsets for k = {k} (Time: {k_time:.2f}s)"
            )

            # Clean up broadcast variable
            broadcast_candidates.unpersist()

            k += 1

        print("\n***** Frequent Itemsets *****")
        for itemset, count in l_all:
            print(f"{itemset}: {count}")

        freq_itemset = {tuple(sorted(itemset)): count for itemset, count in l_all}

        print("\n--- Generating Association Rules ---")
        rules = self._generate_association_rules(freq_itemset, support_threshold)

        print(
            "\n***********Association rules (X -> Y)   [support, confidence]:***************"
        )
        for i, (x, y, support, confidence) in enumerate(rules, 1):
            print(
                f"{i:>2}. {x} -> {y}   [support={support}, confidence={confidence:.3f}]"
            )

        elapsed = time.time() - start_time
        print(f"\nTotal execution time: {elapsed:.2f} seconds")

        # Cleanup
        baskets.unpersist()
        self.spark.stop()

        return {
            "support": self.support_percentage,
            "confidence": self.confidence_threshold,
            "runtime": elapsed,
            "n_itemsets": len(freq_itemset),
            "n_rules": len(rules),
            "k_timings": k_timings,
        }


if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_directory, "resources", "T10I4D100K.dat")

    frequent_items = FrequentItemSets(
        path=file_path, support_percentage=0.005, confidence_threshold=0.6
    )
    frequent_items.extract_frequent_items()
