"""
Multilingual Query Normalization and Language Identification Module
Designed for Pakistan Public Service NLP (Urdu Script, Roman Urdu, English, Code-Mixed)
"""

import re
import unicodedata
from typing import Dict, List, Tuple


class MultilingualNormalizer:
    """
    Handles Unicode normalization for Urdu script, phonetic/spelling variation collapse
    for Roman Urdu, language and script identification, and token generation.
    """

    # Urdu Unicode mappings
    URDU_CHAR_MAP = {
        # Alif variants
        '\u0622': '\u0627',  # Alef with madda above -> Alef
        '\u0623': '\u0627',  # Alef with hamza above -> Alef
        '\u0625': '\u0627',  # Alef with hamza below -> Alef
        # Yeh variants
        '\u064A': '\u06CC',  # Arabic Yeh -> Urdu Choti Yeh
        '\u0649': '\u06CC',  # Alef Maksura -> Urdu Choti Yeh
        '\u06D0': '\u06D2',  # Kashmiri Yeh -> Bari Yeh
        # Kaf variants
        '\u0643': '\u06a9',  # Arabic Kaf -> Urdu Keheh
        # Heh variants
        '\u0629': '\u06c1',  # Teh Marbuta -> Urdu Goal Heh
        '\u0647': '\u06c1',  # Arabic Heh -> Urdu Goal Heh
        '\u06BE': '\u06be',  # Do-chashmi Heh preserved
        # Waw variants
        '\u0624': '\u0648',  # Waw with Hamza -> Waw
        # Diacritics / Tashkeel to remove
        '\u064B': '',  # Fathatan
        '\u064C': '',  # Dammatan
        '\u064D': '',  # Kasratan
        '\u064E': '',  # Fatha
        '\u064F': '',  # Damma
        '\u0650': '',  # Kasra
        '\u0651': '',  # Shadda
        '\u0652': '',  # Sukun
        '\u0670': '',  # Superscript Alef
        '\u0610': '',  # Sign Sallallahou Alayhe Wassallam
        '\u0611': '',
        '\u0612': '',
        '\u0613': '',
        # Digits (Eastern Arabic and Persian/Urdu)
        '\u06F0': '0', '\u06F1': '1', '\u06F2': '2', '\u06F3': '3', '\u06F4': '4',
        '\u06F5': '5', '\u06F6': '6', '\u06F7': '7', '\u06F8': '8', '\u06F9': '9',
        '\u0660': '0', '\u0661': '1', '\u0662': '2', '\u0663': '3', '\u0664': '4',
        '\u0665': '5', '\u0666': '6', '\u0667': '7', '\u0668': '8', '\u0669': '9',
    }

    # Roman Urdu spelling normalizations
    ROMAN_URDU_LEXICON = {
        # Verbs & auxiliary
        r'\b(hy|hei|hye|he)\b': 'hai',
        r'\b(hyn|hen|hain)\b': 'hain',
        r'\b(thaa|tha|thi|thay|the)\b': 'tha',
        r'\b(hoga|hogi|hongay|hoge)\b': 'hoga',
        r'\b(krna|karna|karein|kren)\b': 'karna',
        r'\b(krwana|krana|karwayen|karvana|karwana)\b': 'karwana',
        r'\b(chahye|chaheye|chaheay|chahiay|chahiye|cheye|chahie)\b': 'chahiye',
        r'\b(chotay|chota|choti|chote)\b': 'chotay',
        r'\b(baray|bara|bari|bare)\b': 'baray',
        r'\b(bachay|bacha|bache|bachi|bachon)\b': 'bachay',
        r'\b(walay|wale|wali|wala|walo)\b': 'walay',
        r'\b(kitna|kitnay|kitne|kitni)\b': 'kitna',
        r'\b(konsa|kosa|kounsa|konsi|konse)\b': 'konsa',
        r'\b(kya|kia|kyah)\b': 'kya',
        r'\b(kon|kaun)\b': 'kon',
        r'\b(kahan|kidhar|kdr)\b': 'kahan',
        r'\b(kaise|kesay|kese|kaisy)\b': 'kaise',
        r'\b(zaruri|zaroori|zarori)\b': 'zaroori',
        r'\b(zaroorat|zarurat)\b': 'zaroorat',
        r'\b(lagta|lagte|lagti|lagtay)\b': 'lagta',
        r'\b(din|days|rooz)\b': 'days',
        # Domain terms: Passport & Immigration
        r'\b(pasport|pasprt|passprt|pasporte)\b': 'passport',
        r'\b(renu|renue|reneww|renewed|navikaran)\b': 'renew',
        r'\b(renewal|renwal|renual)\b': 'renewal',
        r'\b(documnt|documnts|dastavez|dastavezat|dastawaiz|dokumnt)\b': 'documents',
        r'\b(nid|cnic|nicop|bay form|b form|form-b|b-form|crc)\b': 'cnic',
        r'\b(police report|fir|raport|complaint)\b': 'police_report',
        r'\b(gum|kho gaya|khogaya|chori|lost)\b': 'lost',
        r'\b(fasttrack|fast-track|fatstrack|tatkal)\b': 'fast_track',
        r'\b(urgnt|urgt|emergency)\b': 'urgent',
        r'\b(nrml|ordinary|aam)\b': 'normal',
        r'\b(gov|govt|sarkari)\b': 'government',
        r'\b(mulk|country|bahir|foreign)\b': 'foreign',
        r'\b(dual national|dual citizenship|dohri shehriyat)\b': 'dual_nationality',
        r'\b(fee|fees|charges|paise|pesay|rupees|rs)\b': 'fee',
        r'\b(office|daftr|daftar|center|centre)\b': 'office',
    }

    # Frequent Roman Urdu indicator stopwords
    ROMAN_URDU_STOPWORDS = {
        'hai', 'hain', 'ka', 'ke', 'ki', 'ko', 'se', 'par', 'per', 'mein', 'me',
        'kya', 'kia', 'kon', 'konsa', 'konsi', 'kaise', 'kese', 'kahan', 'kyun',
        'chahiye', 'karwana', 'krwana', 'karna', 'krna', 'hoga', 'hogi', 'tha',
        'aur', 'ya', 'agar', 'to', 'bhi', 'ab', 'jab', 'tak', 'liye', 'walay',
        'wala', 'wali', 'bache', 'bachay', 'mera', 'meri', 'mere', 'apna', 'apni',
        'unka', 'unki', 'hota', 'hoti', 'hotay', 'karo', 'karen', 'bataen', 'batao',
        'den', 'dein', 'sakta', 'sakti', 'saktay', 'kitna', 'kitni', 'kitnay'
    }

    # Urdu Script stopwords
    URDU_STOPWORDS = {
        'کے', 'کی', 'کو', 'کا', 'سے', 'میں', 'پر', 'ہے', 'ہیں', 'تھا', 'تھی',
        'تھے', 'اور', 'یا', 'اگر', 'تو', 'بھی', 'اب', 'جب', 'تک', 'لیے', 'لئے',
        'والے', 'والا', 'والی', 'بچے', 'میرا', 'میری', 'میرے', 'اپنا', 'اپنے',
        'کیا', 'کون', 'کونسا', 'کونسی', 'کیسے', 'کہاں', 'کیوں', 'چاہیے', 'کرنا',
        'کرانا', 'ہوگا', 'ہوگی', 'ہوتے', 'ہوتا', 'ہوتی', 'سکتا', 'سکتی', 'کتنا',
        'کتنے', 'کتنی', 'درکار', 'شامل', 'بعد', 'پہلے', 'اس', 'ان', 'وہ', 'یہ'
    }

    @staticmethod
    def is_urdu_script(text: str) -> bool:
        """Check if text contains predominantly Urdu / Arabic script characters."""
        urdu_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))
        return urdu_chars > 0 and urdu_chars >= latin_chars

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detects query language:
        - 'ur': Urdu script
        - 'roman_ur': Roman Urdu
        - 'en': English
        - 'code_mixed': Mixed English + Urdu / Roman Urdu
        """
        text_clean = text.strip()
        if not text_clean:
            return 'en'

        # Check for Urdu script
        urdu_matches = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text_clean))
        latin_words = [w.lower() for w in re.findall(r'[a-zA-Z]+', text_clean)]

        if urdu_matches > 0:
            if latin_words:
                return 'code_mixed'
            return 'ur'

        # Analyze Latin text: count Roman Urdu words vs English words
        if not latin_words:
            return 'en'

        roman_urdu_hits = sum(1 for w in latin_words if w in MultilingualNormalizer.ROMAN_URDU_STOPWORDS)
        total_words = len(latin_words)
        ratio = roman_urdu_hits / total_words

        if ratio >= 0.20 or (roman_urdu_hits >= 2 and total_words <= 6):
            # Check if there are strong English markers too
            en_procedural = sum(1 for w in latin_words if w in {'required', 'documents', 'apply', 'procedure', 'applicant', 'validity', 'renewal', 'appointment'})
            if en_procedural >= 2 and roman_urdu_hits >= 2:
                return 'code_mixed'
            return 'roman_ur'

        return 'en'

    @classmethod
    def normalize_urdu_script(cls, text: str) -> str:
        """Normalize Urdu Unicode variants and strip diacritics."""
        # Normalize NFKD
        text = unicodedata.normalize('NFKD', text)

        # Character replacement
        chars = []
        for ch in text:
            chars.append(cls.URDU_CHAR_MAP.get(ch, ch))
        text = ''.join(chars)

        # Remove zero-width non-joiner / joiner and formatting chars
        text = re.sub(r'[\u200c\u200d\u200e\u200f\ufeff]', ' ', text)
        # Collapse punctuation & multiple whitespace
        text = re.sub(r'[۔،؟«»"\'\(\)\{\}\[\]\-_:;/\\]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def normalize_roman_urdu(cls, text: str) -> str:
        """Normalize Roman Urdu spelling variations, elongation, and phonetic variants."""
        text = text.lower().strip()

        # Remove character elongation: e.g. "paspooort" -> "pasport", "chahiiye" -> "chahiye"
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', text)

        # Apply specific lexicon rules
        for pattern, replacement in cls.ROMAN_URDU_LEXICON.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def normalize(cls, query: str) -> Tuple[str, str]:
        """
        Main normalization method.
        Returns:
            normalized_query (str): Cleaned and normalized text
            lang (str): Detected language ('ur', 'roman_ur', 'en', 'code_mixed')
        """
        lang = cls.detect_language(query)

        if lang == 'ur':
            cleaned = cls.normalize_urdu_script(query)
        elif lang in ('roman_ur', 'code_mixed'):
            cleaned = cls.normalize_roman_urdu(query)
        else:
            # English
            cleaned = query.lower().strip()
            # Collapse elongation
            cleaned = re.sub(r'([a-zA-Z])\1{2,}', r'\1', cleaned)
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned, lang

    @classmethod
    def tokenize_for_retrieval(cls, text: str, lang: str = 'en') -> List[str]:
        """
        Produces lexical tokens, including character 3-grams for subword matching
        robust to spelling variations in Roman Urdu and Urdu.
        """
        clean_text, detected_lang = cls.normalize(text)
        words = clean_text.split()

        # Remove language-specific stopwords
        filtered_words = []
        if detected_lang == 'ur':
            filtered_words = [w for w in words if w not in cls.URDU_STOPWORDS and len(w) > 1]
        elif detected_lang in ('roman_ur', 'code_mixed'):
            filtered_words = [w for w in words if w not in cls.ROMAN_URDU_STOPWORDS and len(w) > 1]
        else:
            en_stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'for', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'by', 'do', 'does', 'can', 'should', 'what', 'which', 'who', 'how'}
            filtered_words = [w for w in words if w.lower() not in en_stops and len(w) > 1]

        if not filtered_words:
            filtered_words = words

        # In addition to full words, add character 3-grams for Roman Urdu / English to catch typos like 'pasport' / 'renu'
        tokens = list(filtered_words)
        if detected_lang in ('roman_ur', 'code_mixed', 'en'):
            for w in filtered_words:
                if len(w) >= 4:
                    tokens.extend([f"#{ng}" for ng in cls.generate_subword_ngrams(w, n=3)])

        return tokens

    @classmethod
    def generate_subword_ngrams(cls, word: str, n: int = 3) -> List[str]:
        """Generates subword character n-grams from a word."""
        if len(word) < n:
            return []
        return [word[i:i+n] for i in range(len(word) - n + 1)]

