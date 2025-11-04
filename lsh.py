import hashlib
from collections import defaultdict


class LSH:
    # Locality-Sensitive Hashing for finding similar documents
    # Uses banding technique: divide signature into bands and hash each band

    def __init__(self, threshold=0.8, num_bands=None, num_rows_per_band=None):
        self.threshold = threshold

        # Set default bands and rows if not provided
        # For threshold 0.8, use 20 bands with 5 rows each (for signature length 100)
        if num_bands is None or num_rows_per_band is None:
            self.num_bands = 20
            self.num_rows_per_band = 5
        else:
            self.num_bands = num_bands
            self.num_rows_per_band = num_rows_per_band

    def _hash_band(self, band):
        # Hash a band to a string
        band_str = ",".join(map(str, band))
        return hashlib.md5(band_str.encode("utf-8")).hexdigest()

    def find_candidate_pairs(self, signatures, doc_names=None):
        # Find candidate pairs using LSH
        # Documents with matching bands are candidates

        if doc_names is None:
            doc_names = [f"doc_{i}" for i in range(len(signatures))]

        if len(signatures) == 0:
            return set()

        # Check signature length
        sig_len = len(signatures[0])
        needed = self.num_bands * self.num_rows_per_band
        if sig_len < needed:
            raise ValueError(f"Need signature length >= {needed}, got {sig_len}")

        # Buckets: band_hash -> list of document indices
        buckets = defaultdict(list)
        candidate_pairs = set()

        # For each document, divide signature into bands
        for doc_idx, signature in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.num_rows_per_band
                end = start + self.num_rows_per_band

                # Get this band
                band = signature[start:end]

                # Hash the band
                band_hash = self._hash_band(band)

                # Add to bucket
                bucket_key = f"{band_idx}_{band_hash}"
                buckets[bucket_key].append(doc_idx)

        # Documents in same bucket are candidates
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                import itertools

                for doc1, doc2 in itertools.combinations(bucket_docs, 2):
                    if doc1 < doc2:
                        candidate_pairs.add((doc1, doc2))
                    else:
                        candidate_pairs.add((doc2, doc1))

        return candidate_pairs

    def find_similar_pairs(self, signatures, doc_names=None, verify=True):
        # Find similar pairs using LSH
        # If verify=True, check actual similarity and filter by threshold
        from compareSignatures import CompareSignatures

        if doc_names is None:
            doc_names = [f"doc_{i}" for i in range(len(signatures))]

        # Get candidate pairs
        candidate_pairs = self.find_candidate_pairs(signatures, doc_names)

        if not verify:
            return [(d1, d2, 1.0) for d1, d2 in candidate_pairs]

        # Verify candidates
        similar_pairs = []
        for doc1_idx, doc2_idx in candidate_pairs:
            similarity = CompareSignatures.similarity(
                signatures[doc1_idx], signatures[doc2_idx]
            )
            if similarity >= self.threshold:
                similar_pairs.append((doc1_idx, doc2_idx, similarity))

        return similar_pairs
