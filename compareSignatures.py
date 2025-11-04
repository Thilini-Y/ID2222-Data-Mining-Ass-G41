class CompareSignatures:
    # Compare two minHash signatures
    # Similarity = fraction of positions where they agree

    @staticmethod
    def similarity(signature1, signature2):
        # Compute similarity between two signatures
        if len(signature1) != len(signature2):
            raise ValueError("Signatures must have same length")

        if len(signature1) == 0:
            return 0.0

        # Count how many positions match
        matches = 0
        for i in range(len(signature1)):
            if signature1[i] == signature2[i]:
                matches += 1

        # Return fraction of matches
        return matches / len(signature1)

    @staticmethod
    def similarity_batch(signatures):
        # Compare all pairs of signatures
        import itertools

        results = []

        for sig1, sig2 in itertools.combinations(enumerate(signatures), 2):
            idx1, s1 = sig1
            idx2, s2 = sig2
            sim = CompareSignatures.similarity(s1, s2)
            results.append((idx1, idx2, sim))

        return results
