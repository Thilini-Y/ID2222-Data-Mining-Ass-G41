import random


class MinHashing:
    # Builds minHash signature from a set of shingles
    # Uses random hash functions to create signatures

    def __init__(self, n=100, seed=42):
        # n = number of hash functions (signature length)
        self.n = n
        random.seed(seed)

        # Create n random hash functions: (a*x + b) mod prime
        self.prime = 2**31 - 1  # large prime number
        self.hash_params = []

        for _ in range(n):
            a = random.randint(1, self.prime - 1)
            b = random.randint(0, self.prime - 1)
            self.hash_params.append((a, b))

    def _hash_function(self, x, a, b):
        # Hash function: (a * x + b) mod prime
        return (a * x + b) % self.prime

    def compute_signature(self, shingles):
        # Compute minHash signature for a set of shingles
        # For each hash function, find the minimum hash value
        if not shingles:
            return [self.prime] * self.n

        signature = []

        for a, b in self.hash_params:
            min_hash = self.prime  # start with max value

            # find minimum hash across all shingles
            for shingle in shingles:
                hash_value = self._hash_function(shingle, a, b)
                if hash_value < min_hash:
                    min_hash = hash_value

            signature.append(min_hash)

        return signature

    def compute_signature_batch(self, shingles_list):
        # Compute signatures for multiple documents
        return [self.compute_signature(shingles) for shingles in shingles_list]
