from typing import List

class CompareSignatures:
    @staticmethod
    def similarity(signature_1: List[int], signature_2: List[int]) -> float:
        if not signature_1 or not signature_2 or len(signature_1) != len(signature_2):
            return 0.0
        agree_signatures = sum(1 for a, b in zip(signature_1, signature_2) if a == b)
        return agree_signatures / len(signature_1)
