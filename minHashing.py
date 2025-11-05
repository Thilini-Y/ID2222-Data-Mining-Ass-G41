import random
from typing import List, Set, Tuple
from pyspark import RDD


class MinHashing:
    #take 4294967311 as the next prime higher than 2^32
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
        if not shingles:
            return [prime] * self.num_perm

        signature = [prime] * self.num_perm
        for x in shingles:
            for i, (a, b) in enumerate(hash_params):
                hx = (a * x + b) % prime
                if hx < signature[i]:
                    signature[i] = hx
        return signature

    def compute_signatures_rdd(self, shingled_rdd: RDD) -> RDD:
        broadcast_hash_params = shingled_rdd.context.broadcast(self.hash_params)
        prime = self.prime

        return shingled_rdd.mapValues(
            lambda shingles: self.compute_for_doc(shingles, broadcast_hash_params.value, prime)
        )
