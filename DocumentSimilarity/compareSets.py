class CompareSets:
    @staticmethod
    def jaccard_similarity(set1: set[int], set2: set[int]) -> float:
        # Compute Jaccard similarity between two sets of shingles
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union
