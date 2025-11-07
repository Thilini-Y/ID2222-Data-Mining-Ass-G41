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
        # Combine band index and band values into a string, then hash with MD5
        # MD5 ensures a consistent and uniform bucket assignment
        band_str = f"{band_index}-{'-'.join(map(str, band_values))}"
        return int(hashlib.md5(band_str.encode("utf-8")).hexdigest(), 16)

    # LOCAL (non-Spark) version: find candidate pairs from signatures
    def find_candidates(self, signatures: Dict[str, List[int]]) -> Set[Tuple[str, str]]:
        """
        Local (non-Spark) candidate generation.
        :param signatures: dict of {doc_id: minhash_signature}
        :return: set of (doc1, doc2) pairs likely to be similar
        """
        if not signatures:
            return set()

        # Check that the signature length matches b * r
        num_perm = len(next(iter(signatures.values())))
        if num_perm != self.bands * self.rows_per_band:
            raise ValueError(
                f"Signature length {num_perm} must equal bands*rows_per_band "
                f"({self.bands * self.rows_per_band})"
            )

        # Bucket documents based on each band’s hash
        # 'buckets' maps a hash key to the list of document IDs that fall into it
        buckets = {}

        # Step 1: Split each signature into bands and hash each band to a bucket
        for doc_id, sig in signatures.items():
            for b in range(self.bands):
                # Define the range of rows that belong to this band
                start = b * self.rows_per_band
                end = start + self.rows_per_band

                # Extract this band’s portion of the signature
                band_tuple = tuple(sig[start:end])

                # Hash the band tuple into a bucket ID
                bucket_key = self._hash_band(band_tuple, b)

                # Add the document to the bucket
                buckets.setdefault(bucket_key, []).append(doc_id)

        # Generate unique candidate pairs
        candidates = set()
        for docs in buckets.values():
            if len(docs) > 1:
                # Generate all combinations of documents in this bucket
                # Example: if bucket contains [doc1, doc3, doc7], produce (doc1, doc3), (doc1, doc7), (doc3, doc7)
                for pair in combinations(sorted(docs), 2):
                    candidates.add(pair)
        return candidates

    # DISTRIBUTED (Spark) version: find candidate pairs using RDD operations
    def find_candidates_rdd(self, signatures_rdd: RDD) -> RDD:
        """
        Spark version — find candidate pairs directly in distributed mode.
        :param signatures_rdd: RDD[(doc_id, signature_list)]
        :return: RDD[(doc1, doc2)]
        """
        bands = self.bands
        rows_per_band = self.rows_per_band

        # Map step: Split each document’s signature into bands
        def split_into_bands(item):
            doc_id, sig = item
            num_perm = len(sig)

            # Skip invalid signatures (wrong length)
            if num_perm != bands * rows_per_band:
                return []

            output = []
            for b in range(bands):
                start = b * rows_per_band
                end = start + rows_per_band
                band_tuple = tuple(sig[start:end])

                # Hash band to produce bucket key
                bucket_key = int(
                    hashlib.md5(f"{b}-{band_tuple}".encode("utf-8")).hexdigest(), 16
                )

                # Emit ((band_index, bucket_key), doc_id)
                output.append(((b, bucket_key), doc_id))
            return output

        # Group by bucket: documents that share the same bucket are candidates
        band_buckets = signatures_rdd.flatMap(split_into_bands).groupByKey()

        # Reduce step: Generate candidate pairs from each bucket
        def generate_pairs(bucket):
            _, docs = bucket
            docs = sorted(set(docs))  # ensure unique doc IDs
            pairs = []

            # Produce all unique pairs of documents that share this bucket
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    pairs.append((docs[i], docs[j]))
            return pairs

        # FlatMap over all buckets to produce pairs, then remove duplicates
        return band_buckets.flatMap(generate_pairs).distinct()
