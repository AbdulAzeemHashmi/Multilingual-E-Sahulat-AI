import unittest
import sys
import os

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.normalizer import MultilingualNormalizer


class TestMultilingualNormalizer(unittest.TestCase):

    def test_language_detection(self):
        self.assertEqual(MultilingualNormalizer.detect_language("پاسپورٹ کی تجدید"), "ur")
        self.assertEqual(MultilingualNormalizer.detect_language("Passport renewal procedure"), "en")
        self.assertEqual(MultilingualNormalizer.detect_language("Passport renew karwana hai"), "roman_ur")

    def test_urdu_script_cleaning(self):
        # Urdu text with mixed digits and diacritics
        raw = "پاسپورٹ نمبر ۱۲۳۴۵"
        normalized = MultilingualNormalizer.normalize_urdu_script(raw)
        self.assertIn("12345", normalized)

    def test_roman_urdu_normalization(self):
        # Spelling variations collapse to canonical representations
        v1 = MultilingualNormalizer.normalize_roman_urdu("pasport renu karwana hy")
        v2 = MultilingualNormalizer.normalize_roman_urdu("passport renew karwana hai")
        self.assertIn("passport", v1)
        self.assertIn("renew", v1)
        self.assertIn("passport", v2)
        self.assertIn("renew", v2)

    def test_subword_ngrams(self):
        tokens = MultilingualNormalizer.generate_subword_ngrams("passport", n=3)
        self.assertIn("pas", tokens)
        self.assertIn("ass", tokens)
        self.assertIn("ssp", tokens)


if __name__ == "__main__":
    unittest.main()
