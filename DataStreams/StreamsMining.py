import random
import sys
import time
from collections import defaultdict


# PART 1: Reservoir Sampling
def sample_edge(n, s, sample_set):
    #Decides whether to insert the n th edge into the sample of size s.
    if n <= s:
        # add first s items to sample
        return True, None
    else:
        # Reservoir Sampling
        # Probability of keeping the new item is s / n
        if random.random() < (s / n):
            # Select a uniform random edge to remove from s
            removed_edge = random.choice(list(sample_set))
            return True, removed_edge
        else:
            return False, None



# PART 2: Triest Based algorith
class TriestBase:
    def __init__(self, M):
        self.M = M                          # Reservoir size
        self.S = set()                      # Sampled edges
        self.neighbour = defaultdict(set)   # Adjacency inside S
        self.processed_edges = 0            # Number of processed edges

        self.global_triangle = 0            # Global triangle count
        self.local_triangle = defaultdict(int)

    def get_shared_neighbors(self, u, v):
        if u not in self.neighbour or v not in self.neighbour:
            return set()
        return self.neighbour[u].intersection(self.neighbour[v])

    def update_counters(self, op, shared, u, v):
        sign = 1 if op == "+" else -1
        c = len(shared)
        if c == 0:
            return

        # Update global
        self.global_triangle += sign * c

        # Update local counters
        self.local_triangle[u] += sign * c
        self.local_triangle[v] += sign * c

        for w in shared:
            self.local_triangle[w] += sign

    def process_edge(self, u, v):
        self.processed_edges += 1

        edge = tuple(sorted((u, v)))

        # Reservoir sampling decision
        insert, removed = sample_edge(self.processed_edges, self.M, self.S)

        if not insert:
            return

        # old edge is removed & update counters
        if removed is not None:
            ru, rv = removed
            shared_r = self.get_shared_neighbors(ru, rv)
            self.update_counters("-", shared_r, ru, rv)

            self.S.remove(removed)
            self.neighbour[ru].remove(rv)
            self.neighbour[rv].remove(ru)

        # 3. Insert the new edge & update counters
        shared_new = self.get_shared_neighbors(u, v)
        self.update_counters("+", shared_new, u, v)

        # Add to sample
        self.S.add(edge)
        self.neighbour[u].add(v)
        self.neighbour[v].add(u)

    def get_estimation(self):
        """
        The TRIÈST-BASE estimator:
            t * (t-1) * (t-2) / (M * (M-1) * (M-2))
        """
        t = self.processed_edges
        M = self.M

        if t <= 2 or M <= 2:
            return self.global_triangle

        scale = (t * (t - 1) * (t - 2)) / (M * (M - 1) * (M - 2))

        return int(self.global_triangle * scale)



# PART 3: Triest Improved algorith
class TriestImproved:
    def __init__(self, sample_size):
        self.sample_size = sample_size  # reservoir (sample) size
        self.sample_edges = set()
        self.neighbors = defaultdict(set)
        self.processed_edges = 0
        self.global_est = 0.0   # global triangle estimation counter
        self.local_est = defaultdict(float)     # per node triangle estimates

    # get common neighbours of u and v
    def get_shared_neighbors(self, u, v):
        if u not in self.neighbors or v not in self.neighbors:
            return set()
        return self.neighbors[u].intersection(self.neighbors[v])

    '''calculates how much each detected triangle should be multiplied to 
    compensate for missing edges as we keep sample'''
    def get_weight(self):
        current_processed_edges = self.processed_edges
        if current_processed_edges <= self.sample_size:
            return 1.0
        else:
            num = (current_processed_edges - 1) * (current_processed_edges - 2)
            den = self.sample_size * (self.sample_size - 1)
            return max(1.0, num / den)

    def process_edge(self, u, v):
        self.processed_edges += 1
        edge = tuple(sorted((u, v)))

        shared_neighbours = self.get_shared_neighbors(u, v)
        if shared_neighbours:
            weight = self.get_weight()
            count = len(shared_neighbours)
            self.global_est += weight * count
            self.local_est[u] += weight * count
            self.local_est[v] += weight * count
            for node in shared_neighbours:
                self.local_est[node] += weight

        # Reservoir Logic
        insert, removed = sample_edge(self.processed_edges, self.sample_size, self.sample_edges)

        if insert:
            if removed:
                remove_u, remove_v = removed
                self.sample_edges.remove(removed)
                self.neighbors[remove_u].remove(remove_v)
                self.neighbors[remove_v].remove(remove_u)
                if not self.neighbors[remove_u]:
                    del self.neighbors[remove_u]
                if not self.neighbors[remove_v]:
                    del self.neighbors[remove_v]

            # Insert new edge
            self.sample_edges.add(edge)
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)

    def get_estimation(self):
        return int(self.global_est)


# Naive in memory exact counter for verification
class ExactCounter:

    def __init__(self):
        self.neighbours = defaultdict(set)
        self.count = 0

    def process_edge(self, u, v):
        # Check intersection
        if u in self.neighbours and v in self.neighbours:
            shared = self.neighbours[u].intersection(self.neighbours[v])
            self.count += len(shared)

        # Add edge
        self.neighbours[u].add(v)
        self.neighbours[v].add(u)


def read_dataset(filepath):
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("%"):
                continue
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u != v:
                        yield u, v
                except ValueError:
                    continue


def main():
    dataset_path = "DataSet/facebook_combined.txt"
    sample_size = 10000  # Sample size

    # Check if file exists
    import os

    if not os.path.exists(dataset_path):
        print(f"Error: File '{dataset_path}' not found.")
        print(
            "Please download a dataset (e.g., from SNAP 'Web graphs') and place it in this folder."
        )
        return

    print(f"--- Processing {dataset_path} with M={sample_size} ---")

    # Initialize Algorithms
    triest_base = TriestBase(sample_size)
    triest_algo = TriestImproved(sample_size)
    exact_algo = ExactCounter()

    start_time = time.time()

    # Stream Processing Loop
    edge_count = 0
    for u, v in read_dataset(dataset_path):
        edge_count += 1

        # Run Algorithms
        triest_base.process_edge(u, v)
        triest_algo.process_edge(u, v)
        exact_algo.process_edge(u, v)

        if edge_count % 10000 == 0:
            sys.stdout.write(f"\rProcessed {edge_count} edges...")
            sys.stdout.flush()

    print(f"\n\nStream ended. Total edges (T): {edge_count}")
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

    true_count = exact_algo.count
    triest_base_est = triest_base.get_estimation()
    triest_estimation = triest_algo.get_estimation()

    def calculate_mape(est, true_val):
        if true_val == 0:
            return 0.0
        return abs(true_val - est) / true_val

    print("-" * 50)
    print("FINAL RESULTS")
    print("-" * 50)
    print(f"{'Algorithm':<15} | {'Triangle Count':<15} | {'Error (MAPE)':<10}")
    print("-" * 50)
    print(f"{'Ground Truth':<15} | {true_count:<15} | {'0.0%':<10}")
    print(
        f"{'TRIEST-BASE':<15} | {triest_base_est:<15} | {calculate_mape(triest_base_est, true_count):.2%}"
    )
    print(
        f"{'TRIEST-IMPR':<15} | {triest_estimation:<15} | {calculate_mape(triest_estimation, true_count):.2%}"
    )
    print("-" * 50)


if __name__ == "__main__":
    main()