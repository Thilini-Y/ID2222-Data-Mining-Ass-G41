from pyspark.sql import SparkSession
from shingling import Shingling


class SparkShinglingApp:
    """
    Main driver class that uses Apache Spark to perform shingling in parallel.
    """

    def __init__(self, k=9):
        self.k = k
        self.shingler = Shingling(k)
        self.spark = SparkSession.builder \
            .appName("ShinglingWithSpark") \
            .master("local[*]") \
            .getOrCreate()
        self.sc = self.spark.sparkContext

    def run(self, documents):
        """
        Perform shingling on all documents in parallel using Spark RDDs.
        """
        print(f"=== Shingling using Apache Spark (k={self.k}) ===")

        # Get local variables (avoid serializing 'self')
        shingler = self.shingler  # safe to use
        sc = self.sc

        # Parallelize documents
        rdd = sc.parallelize(list(documents.items()))

        # Apply shingling (no self inside lambda)
        shingled_rdd = rdd.map(lambda x: (x[0], shingler.create_shingles(x[1])))

        # Collect results
        results = shingled_rdd.collect()

        for doc_id, shingles in results:
            print(f"\nDocument: {doc_id}")
            print(f"Number of shingles: {len(shingles)}")
            print("Sample hashed shingles (first 5):")
            for s in list(shingles)[:5]:
                print(f"  {s}")

        self.spark.stop()


if __name__ == "__main__":
    documents = {
        "doc1": "The quick brown fox jumps over the lazy dog.",
        "doc2": "The fast brown fox leaps over the lazy hound.",
        "doc3": "Python and Spark are powerful tools for big data processing."
    }

    app = SparkShinglingApp(k=9)
    app.run(documents)
