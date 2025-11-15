import re
import hashlib
from typing import Set


class Shingling:
    def __init__(self, k: int = 10):
        self.k = k

    @staticmethod
    def _normalize(text: str) -> str:
        # covert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', ' ', text)
        return text.strip()

    @staticmethod
    def _get_hash(s: str) -> int:
        # convert text -> hash → integer using md5
        return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

    def create_shingles(self, text: str) -> Set[int]:
        normalized_text = self._normalize(text)
        if len(normalized_text) < self.k:
            return set()

        shingles = {
            self._get_hash(normalized_text[i:i + self.k])
            for i in range(len(normalized_text) - self.k + 1)
        }
        return shingles
