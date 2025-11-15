import random
from typing import List, Set, Tuple
from pyspark import RDD
import numpy as np


class MinHashing:
    # take 4294967311 as the next prime higher than 2^32
    def __init__(self, num_perm: int = 100, seed: int = 1, prime: int = 4294967311):
        if num_perm <= 0:
            raise ValueError("num_perm must be > 0")
        self.num_perm = num_perm
        self.prime = prime
        self.rand = random.Random(seed)

        # generate hash function coefficients (a, b)
        self.hash_params: List[Tuple[int, int]] = [
            (self.rand.randrange(1, prime - 1), self.rand.randrange(1, prime - 1))
            for _ in range(num_perm)
        ]

    def compute_for_doc(self, shingles: Set[int], hash_params: List[Tuple[int, int]], prime: int) -> List[int]:
        # compute minhash signatures
        if not shingles:
            return [prime] * self.num_perm

        # reduce all shingle values modulo prime to avoid OverflowError
        shingles_arr = np.array([x % prime for x in shingles], dtype=np.uint64)
        a = np.array([h[0] for h in hash_params], dtype=np.uint64)
        b = np.array([h[1] for h in hash_params], dtype=np.uint64)

        # (a*x + b) % c for all hash functions and shingles at once
        hashes = (a[:, None] * shingles_arr[None, :] + b[:, None]) % prime

        # Take the minimum hash per permutation
        signature = np.min(hashes, axis=1)
        return signature.tolist()

    def compute_signatures_rdd(self, shingled_rdd: RDD) -> RDD:
        #compute minhash signatures for each document parallely using spark
        broadcast_hash_params = shingled_rdd.context.broadcast(self.hash_params)
        prime = self.prime

        # spark worker computes MinHash for its assigned document subset
        return shingled_rdd.mapValues(
            lambda shingles: self.compute_for_doc(shingles, broadcast_hash_params.value, prime)
        )
