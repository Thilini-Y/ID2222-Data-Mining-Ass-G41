import hashlib
from itertools import combinations
from typing import Dict, List, Set, Tuple
from pyspark import RDD


class LSH:
    """
    Locality-Sensitive Hashing (LSH) using the banding technique.
    Given MinHash signatures, groups documents into buckets to quickly
    find candidate pairs of potentially similar documents.
    """

    def __init__(self, bands: int, rows_per_band: int):
        if bands <= 0 or rows_per_band <= 0:
            raise ValueError("Both bands and rows_per_band must be > 0")
        self.bands = bands
        self.rows_per_band = rows_per_band

    @staticmethod
    def _hash_band(band_values: Tuple[int, ...], band_index: int) -> int:
        """Hash a band tuple and band index to a deterministic integer key."""
        band_str = f"{band_index}-{'-'.join(map(str, band_values))}"
        return int(hashlib.sha1(band_str.encode("utf-8")).hexdigest(), 16)

    def find_candidates(self, signatures: Dict[str, List[int]]) -> Set[Tuple[str, str]]:
        """
        Local (non-Spark) candidate generation.
        :param signatures: dict of {doc_id: minhash_signature}
        :return: set of (doc1, doc2) pairs likely to be similar
        """
        if not signatures:
            return set()

        num_perm = len(next(iter(signatures.values())))
        if num_perm != self.bands * self.rows_per_band:
            raise ValueError(
                f"Signature length {num_perm} must equal bands*rows_per_band "
                f"({self.bands * self.rows_per_band})"
            )

        # Bucket documents based on each band’s hash
        buckets = {}
        for doc_id, sig in signatures.items():
            for b in range(self.bands):
                start = b * self.rows_per_band
                end = start + self.rows_per_band
                band_tuple = tuple(sig[start:end])
                bucket_key = self._hash_band(band_tuple, b)
                buckets.setdefault(bucket_key, []).append(doc_id)

        # Generate unique candidate pairs
        candidates = set()
        for docs in buckets.values():
            if len(docs) > 1:
                for pair in combinations(sorted(docs), 2):
                    candidates.add(pair)
        return candidates

    def find_candidates_rdd(self, signatures_rdd: RDD) -> RDD:
        """
        Spark version — find candidate pairs directly in distributed mode.
        :param signatures_rdd: RDD[(doc_id, signature_list)]
        :return: RDD[(doc1, doc2)]
        """
        bands = self.bands
        rows_per_band = self.rows_per_band

        def split_into_bands(item):
            doc_id, sig = item
            num_perm = len(sig)
            if num_perm != bands * rows_per_band:
                return []
            output = []
            for b in range(bands):
                start = b * rows_per_band
                end = start + rows_per_band
                band_tuple = tuple(sig[start:end])
                bucket_key = int(
                    hashlib.sha1(f"{b}-{band_tuple}".encode("utf-8")).hexdigest(), 16
                )
                output.append(((b, bucket_key), doc_id))
            return output

        band_buckets = signatures_rdd.flatMap(split_into_bands).groupByKey()

        def generate_pairs(bucket):
            _, docs = bucket
            docs = sorted(set(docs))
            pairs = []
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    pairs.append((docs[i], docs[j]))
            return pairs

        return band_buckets.flatMap(generate_pairs).distinct()
