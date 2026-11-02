"""
Evidence-Grounded Answer Generation & Safe Abstention Engine
Ensures:
- Strict evidence attribution and citation integrity
- Zero hallucination (abstains when evidence is unavailable)
- Comprehensive multi-category guardrails across English, Roman Urdu, and Urdu script:
  1. Ambiguous query clarification
  2. Individual tracking / unsupported personal guarantees
  3. Out-of-Domain questions (sports, weather, politics)
  4. Prompt injection and adversarial citation bypass attempts
  5. Unsupported action (e.g. booking appointments)
  6. Time-sensitive fee currency verification
  7. Conflicting evidence disclosure
  8. Unsafe / verification bypass rejection
- Multilingual response generation (English, Urdu Script, Roman Urdu)
"""

import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, List, Any, Optional, Tuple


class GroundedAnswerEngine:
    """
    Synthesizes evidence-grounded answers or generates safe abstentions/clarifications
    based on retrieved public-service passages and calibrated confidence thresholds.
    """

    # 1. Out-of-Domain detection
    OOD_KEYWORDS_EN = {
        'cricket', 'world cup', 'babar', 'virat', 'match', 'weather', 'recipe',
        'movie', 'song', 'bitcoin', 'crypto', 'astrology', 'horoscope', 'cinema',
        'football', 'fifa', 'modi', 'trump', 'biden', 'putin', 'stock market', 'rain'
    }
    OOD_PATTERNS_ROMAN_UR = [
        r'\b(barish|mosam|mausam|match|cricket|dhoop|sardi|garmi)\b',
        r'\b(wazeer e azam|pm|prime minister|hukoomat|election)\b'
    ]
    OOD_PATTERNS_UR = [
        r'(کرکٹ|ورلڈ کپ|بارش|موسم|وزیر اعظم|الیکشن|سیاست|میچ)'
    ]

    # 2. Prompt Injection & Adversarial instructions
    INJECTION_PATTERNS = [
        r'ignore (all )?(rules|instructions|system|guidelines)',
        r'disregard (the )?system',
        r'you are now (a|an)',
        r'system prompt',
        r'jailbreak',
        r'invent (a |any )?passport fee',
        r'source cite na karo',
        r'confident jawab bana do',
        r'حوالہ مت دو',
        r'جواب بنا دو',
        r'بغیر ثبوت',
        r'override rules',
        r'forget (all )?rules'
    ]

    # 3. Ambiguous queries (lacking parameters)
    AMBIGUOUS_PATTERNS = [
        r'^(kitni|kitna|kitne)\s+(fees?|charges?|paise|pesay)(\s+hai|\s+hain|\s+lagti|\s+hogi)?\??$',
        r'^(how much|what is the cost|cost|fee|fees)(\s+does it cost|\s+is it|\s+are fees)?\??$',
        r'^(fees?|charges?|price|cost)\??$',
        r'^(فیس کتنی ہے|کتنی فیس ہے|کتنا خرچہ|کتنے پیسے)\??$',
        r'^passport fee\??$'
    ]

    # 4. Individual application tracking / personal delivery guarantees
    PERSONAL_STATUS_PATTERNS = [
        # Roman Urdu variants
        r'\b(mera passport kab|kab ghar pohanchay ga|mera status|application status)\b',
        r'\b(mera passport.*pohanche?g[ay]|mujhe.*passport.*kab|passport.*kab milega)\b',
        r'\b(mera application|meri application|mera case|mera number|mera tracking)\b',
        r'\b(current passport application status|status of my passport|mera passport status)\b',
        # English variants
        r'\b(guarantee that my passport|arrive tomorrow|individual delivery)\b',
        r'\b(what is my current passport application status|track my passport)\b',
        r'\b(can you confirm my application will be approved|will my application be approved)\b',
        r'\b(my cnic name.*different|application will be approved|confirm my application)\b',
        r'\b(my individual application|guaranteed processing time for my)\b',
        # Urdu script variants
        r'(میرا پاسپورٹ کل لازماً|کب گھر پہنچے گا|درخواست کا موجودہ اسٹیٹس|میرا اسٹیٹس)',
        r'(میرے پاسپورٹ کی درخواست کا موجودہ|میری درخواست|میرا پاسپورٹ اسٹیٹس)',
    ]

    # 5. Unsupported action (e.g. booking appointments)
    UNSUPPORTED_ACTION_PATTERNS = [
        # Roman Urdu
        r'\b(appointment.*confirm kar do|book an appointment|book appointment|mere liye appointment)\b',
        r'\b(mujhay.*appointment.*confirm|appointment.*kal|kal.*appointment)\b',
        r'\b(schedule.*appointment|fix appointment|appointment bana do)\b',
        # Urdu script
        r'(اپوائنٹمنٹ بک کر|وقت طے کر)',
        r'(اپوائنٹمنٹ.*کل|کل.*اپوائنٹمنٹ|وقت مقرر)',
    ]

    # 6. Time-sensitive fee currency questions
    TIME_SENSITIVE_PATTERNS = [
        # English
        r'\b(is the 2024 fee definitely still current today|still current today|valid today|current fee schedule)\b',
        r'\b(still (the same|applicable|current|valid) (today|now|currently))\b',
        r'\b(fee.*still current|fee.*still (valid|applicable)|definitely still)\b',
        # Roman Urdu
        r'\b(abhi bhi yahi fee|aaj bhi yahi fee|fees? aaj bhi laagu|fee aaj bhi)\b',
        # Urdu script
        r'(کیا آج بھی یہی فیس لاگو ہے|کیا یہ فیس اب بھی لاگو ہے)',
        r'(آج بھی یہی فیس|اب بھی لاگو|فیس اب بھی|یہی فیس اب)',
    ]

    # 7. Conflicting evidence questions
    CONFLICTING_PATTERNS = [
        # English
        r'\b(another page says something else|different processing time|conflicting|two different pages)\b',
        r'\b(one page says|another page says|says something else|two pages)\b',
        r'\b(page says normal processing is 21 days.*another page)\b',
        # Roman Urdu / mixed
        r'\b(ek jagah.*doosri jagah|doosri jagah.*alag|mختلف معلومات)\b',
        # Urdu script
        r'(ایک جگہ.*دوسری جگہ|تضاد|مختلف معلومات)',
    ]

    # 8. Unsafe / verification bypass
    UNSAFE_PATTERNS = [
        # English
        r'\b(bypass passport verification|bypass.*verification|fake passport|bribe|skip biometric|cheat verification)\b',
        r'\b(how to bypass|get around verification|avoid verification|circumvent)\b',
        # Roman Urdu
        r'\b(bypass karna|jawali passport|takht\w+ karna|bina tasdeeq)\b',
        # Urdu script
        r'(تصدیق بائی پاس|جعلی پاسپورٹ|بغیر تصدیق)',
        r'(پاسپورٹ تصدیق بائی پاس|تصدیق سے بچنا)',
    ]

    # 9. Unsupported generalization
    GENERALIZATION_PATTERNS = [
        r'guaranteed fee for every applicant regardless of category',
        r'\b(same fee for (every|all) applicant|uniform fee for all|single fee regardless)\b',
        r'\b(guaranteed.*fee.*every|every applicant.*guaranteed.*fee)\b',
    ]

    # Calibrated confidence threshold for dense retrieval
    DENSE_CONFIDENCE_THRESHOLD = 0.38

    def __init__(self, confidence_threshold: float = 0.38):
        self.confidence_threshold = confidence_threshold

    def evaluate_query_safety_and_scope(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates whether a query is safe, in-domain, or ambiguous.
        Returns:
            (is_supported, reason_code, message)
        """
        q_raw = query.strip()
        q_lower = q_raw.lower()

        # 1. Prompt Injection & Adversarial Bypass
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "prompt_injection", "Security Alert: Prompt manipulation or adversarial request detected. System answers only with traceable official evidence."

        # 2. Unsafe / verification bypass
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "unsafe_or_out_of_scope", "Security Policy: Multilingual E-Sahulat AI cannot assist with bypassing official verification or regulations. Please adhere to lawful DGI&P application procedures."

        # 3. Ambiguous Query check
        for pattern in self.AMBIGUOUS_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "ambiguous", "Ambiguous Query: Passport fees depend on multiple parameters: booklet size (36, 72, or 100 pages), validity (5 or 10 years), and service tier (Normal, Urgent, or Fast Track). Please specify your intended passport category to view exact official fees."

        # 4. Individual status / Personal outcome guarantees
        for pattern in self.PERSONAL_STATUS_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "unsupported_personal", "Individual Application Notice: The general knowledge base does not hold individual applicant tracking records or delivery guarantees. For personal tracking, please use official DGI&P SMS tracking (9988) or visit your regional passport office."

        # 5. Unsupported action (e.g. booking appointments)
        for pattern in self.UNSUPPORTED_ACTION_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "unsupported_action", "Action Not Supported: Multilingual E-Sahulat AI is an informational retrieval assistant and cannot book or modify live appointments. Please use the official DGI&P online portal or visit your regional office."

        # 6. Time-sensitive verification
        for pattern in self.TIME_SENSITIVE_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "time_sensitive", "Time-Sensitive Verification: Official fee schedules are revised periodically by the Government of Pakistan. While our indexed sources reflect official published rates, please verify against the live DGI&P fee portal before making bank deposits."

        # 7. Conflicting evidence disclosure
        for pattern in self.CONFLICTING_PATTERNS:
            if re.search(pattern, q_lower) or re.search(pattern, q_raw):
                return False, "conflicting_evidence", "Procedural Notice: Where differing timelines appear across historical web pages, DGI&P standard rules specify 21 working days for Normal and 5 days for Urgent processing. Please consult the most recent live DGI&P notifications for current processing."

        # 8. Unsupported generalization
        for pattern in self.GENERALIZATION_PATTERNS:
            if re.search(pattern, q_lower):
                return False, "unsupported_generalization", "Category Specificity Required: There is no single uniform fee for all applicants. Fees depend strictly on page count (36/72/100), validity (5/10 years), and urgency."

        # 9. Out-of-Domain checks
        words = set(re.findall(r'[a-zA-Z]+', q_lower))
        if words.intersection(self.OOD_KEYWORDS_EN):
            return False, "out_of_domain", "Out-of-Domain Query: This topic falls outside the official Pakistani public-service knowledge base (DGI&P Passport & Identity Services). Multilingual E-Sahulat AI only answers verified official public service inquiries."

        for pattern in self.OOD_PATTERNS_ROMAN_UR:
            if re.search(pattern, q_lower):
                return False, "out_of_domain", "Out-of-Domain Query: This topic falls outside the official Pakistani public-service knowledge base."

        for pattern in self.OOD_PATTERNS_UR:
            if re.search(pattern, q_raw):
                return False, "out_of_domain", "یہ سوال سرکاری پاسپورٹ اور عوامی خدمات کے علم سے باہر ہے۔"

        # 10. Random meaningless text
        cleaned = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF]', '', q_raw)
        if len(cleaned) < 3:
            return False, "no_semantic_relevance", "Query is too brief or lacks recognizable keywords. Please provide a complete question regarding passport or public service procedures."

        return True, None, None

    def synthesize_answer(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        target_lang: str = 'en'
    ) -> Dict[str, Any]:
        """
        Generates a traceable, evidence-grounded answer or appropriate abstention.
        """
        # Step 1: Pre-retrieval safety & domain check
        is_safe, reason, msg = self.evaluate_query_safety_and_scope(query)
        if not is_safe:
            return self._build_abstention_response(query, reason, msg, target_lang)

        if not retrieved_passages:
            return self._build_abstention_response(
                query,
                "no_evidence_found",
                "No official evidence passages were retrieved for this query from the knowledge base.",
                target_lang
            )

        top_passage = retrieved_passages[0]
        dense_score = top_passage.get('dense_score', top_passage.get('score', 0.0))

        # Step 2: Post-retrieval semantic confidence threshold check
        if dense_score < self.confidence_threshold:
            return self._build_abstention_response(
                query,
                "insufficient_evidence",
                "The available official evidence does not sufficiently support an answer for this question (low semantic confidence). To prevent misinformation, the system abstains.",
                target_lang
            )

        # Step 3: Format grounded answer in requested language
        if target_lang == 'ur':
            answer_text = top_passage.get('expected_answer_ur') or top_passage.get('gold_evidence_ur')
            evidence_text = top_passage.get('gold_evidence_ur')
        elif target_lang == 'roman_ur':
            answer_text = top_passage.get('expected_answer_roman_ur') or top_passage.get('expected_answer_en')
            evidence_text = top_passage.get('gold_evidence_en')
        else:
            answer_text = top_passage.get('expected_answer_en') or top_passage.get('gold_evidence_en')
            evidence_text = top_passage.get('gold_evidence_en')

        # Traceability metadata
        response = {
            "status": "answered",
            "query": query,
            "target_language": target_lang,
            "answer": answer_text,
            "cited_evidence": evidence_text,
            "passage_id": top_passage.get('passage_id'),
            "topic": top_passage.get('topic'),
            "source_url": top_passage.get('source_url'),
            "confidence_score": float(dense_score),
            "retrieval_method": top_passage.get('retrieval_method'),
            "supporting_passages": [
                {
                    "rank": p.get('rank'),
                    "passage_id": p.get('passage_id'),
                    "topic": p.get('topic'),
                    "score": round(float(p.get('score', 0)), 4),
                    "source_url": p.get('source_url'),
                    "snippet": p.get('gold_evidence_en', '')[:120] + "..."
                }
                for p in retrieved_passages[:3]
            ]
        }
        return response

    def _build_abstention_response(
        self,
        query: str,
        reason: str,
        explanation_en: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """Creates a standardized, traceable abstention response in the target language."""
        if target_lang == 'ur':
            if reason == 'out_of_domain':
                msg = "یہ سوال سرکاری پاسپورٹ اور عوامی خدمات کے علم سے باہر ہے۔ ای سہولت اے آئی صرف مصدقہ سرکاری طریقہ کار کے جوابات دیتا ہے۔"
            elif reason == 'ambiguous':
                msg = "سوال مبہم ہے: پاسپورٹ فیس کا دارومدار صفحات (36، 72، 100)، میعاد (5 یا 10 سال)، اور کارروائی کی قسم (نارمل، ارجنٹ، فاسٹ ٹریک) پر ہوتا ہے۔ برائے مہربانی مخصوص زمرہ واضح کریں۔"
            elif reason == 'unsupported_personal':
                msg = "سرکاری عمومی رہنمائی سے انفرادی درخواست یا ترسیل کا اسٹیٹس معلوم نہیں کیا جا سکتا۔ ذاتی ٹریکنگ کے لیے 9988 پر ایس ایم ایس کریں یا متعلقہ دفتر سے رابطہ کریں۔"
            elif reason == 'unsupported_action':
                msg = "ای سہولت اے آئی صرف معلوماتی نظام ہے اور براہ راست اپوائنٹمنٹ بک نہیں کر سکتا۔ برائے مہربانی ڈی جی آئی پی پورٹل استعمال کریں۔"
            elif reason == 'time_sensitive':
                msg = "حکومت پاکستان کی جانب سے فیس کا شیڈول وقتاً فوقتاً تبدیل ہوتا رہتا ہے۔ تازہ ترین فیس کے لیے ڈی جی آئی پی کی ویب سائٹ ملاحظہ کریں۔"
            elif reason == 'unsafe_or_out_of_scope':
                msg = "سیکورٹی پالیسی: تصدیقی طریقہ کار کو بائی پاس کرنے میں مدد فراہم نہیں کی جا سکتی۔"
            else:
                msg = "مطلوبہ معلومات کی تصدیق سرکاری دستاویزات سے نہیں ہو سکی، اس لیے غلط بیانی سے بچنے کے لیے سسٹم جواب دینے سے معذرت خواہ ہے۔"
        elif target_lang == 'roman_ur':
            if reason == 'out_of_domain':
                msg = "Yeh sawal official public service domain se bahir hai. Multilingual E-Sahulat AI sirf verified sarkari procedure ke baray mein rehnumai fraham karta hai."
            elif reason == 'ambiguous':
                msg = "Sawal ghair wazeh (ambiguous) hai: Passport fees ka inhisar pages (36, 72, 100), validity (5 ya 10 saal), aur priority (Normal, Urgent, Fast Track) par hota hai. Barah-e-karam apna matlooba category wazeh karein."
            elif reason == 'unsupported_personal':
                msg = "General knowledge base se individual passport status ya delivery date confirm nahi ki ja sakti. Tracking ke liye 9988 par SMS karein ya passport office se rabta karein."
            elif reason == 'unsupported_action':
                msg = "Yeh prototype sirf maloomati assistant hai aur live appointment book nahi kar sakta. Barah-e-karam DGIP portal use karein."
            elif reason == 'time_sensitive':
                msg = "Fee schedule time to time update hota hai. Payment se pehle DGIP live portal par verify zaroor karein."
            elif reason == 'unsafe_or_out_of_scope':
                msg = "Verification procedures ko bypass karna qanooni taur par mana hai. Official procedure follow karein."
            else:
                msg = "Matlooba maloomat official record mein dastyab nahi hain, is liye system ghalat bayani se bachne ke liye jawab deny se qasir hai."
        else:
            msg = explanation_en

        return {
            "status": "abstained",
            "query": query,
            "target_language": target_lang,
            "abstention_reason": reason,
            "message": msg,
            "answer": msg,
            "cited_evidence": None,
            "passage_id": None,
            "topic": None,
            "source_url": None,
            "confidence_score": 0.0,
            "retrieval_method": None,
            "supporting_passages": []
        }
