import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import MultilingualRetriever


class TestMultilingualRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = MultilingualRetriever()

    def test_corpus_loaded(self):
        self.assertGreaterEqual(len(self.retriever.passages), 30)

    def test_bm25_retrieval(self):
        results = self.retriever.retrieve_lexical("passport renewal documents", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("passage_id", results[0])
        self.assertIn("score", results[0])

    def test_dense_retrieval(self):
        results = self.retriever.retrieve_dense("What is required to renew passport?", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("passage_id", results[0])
        self.assertIn("score", results[0])

    def test_hybrid_retrieval(self):
        results = self.retriever.retrieve_hybrid("Passport renew karwane ke liye documents", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("passage_id", results[0])
        self.assertIn("score", results[0])

    def test_retrieval_traceability_fields(self):
        results = self.retriever.retrieve("What is the fee for 36 pages?", method="hybrid", top_k=1)
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertIn("passage_id", top)
        self.assertIn("source_url", top)
        self.assertIn("topic", top)
        self.assertTrue(top["source_url"].startswith("http"))


if __name__ == "__main__":
    unittest.main()
