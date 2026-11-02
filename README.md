# Multilingual E-Sahulat AI (ملٹی لنگوئل ای سہولت اے آئی)
### Evidence-Grounded Multilingual NLP for Pakistan's Public Services

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Interface-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Zero External APIs](https://img.shields.io/badge/APIs-100%25%20Local%20%2F%20Offline-success.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Multilingual E-Sahulat AI is a sovereign, evidence-grounded multilingual NLP assistant designed for Pakistan's public-service domain (**Directorate General of Immigration & Passports: DGI&P**). It enables citizens to retrieve authoritative public-service procedures using **Urdu script**, **Roman Urdu**, **English**, and **code-mixed** queries. It guarantees passage-level traceability with verified official links and reliably abstains when evidence is unavailable, conflicting, or out-of-domain.

---

## 🏆 Hackathon Challenge Details

* **Challenge:** Hackathon Challenge: Multilingual E-Sahulat AI (Evidence-Grounded Multilingual NLP for Pakistan)
* **Domain Focus:** Directorate General of Immigration & Passports (DGI&P), Ministry of Interior, Government of Pakistan
* **Time Limit:** 3 to 3.5 hours
* **Team Size:** 2 to 3 participants
* **Resource Constraints:** Free and open-source tools, models, and datasets. Models must be loaded locally in custom code. External API calls (Gemini, ChatGPT, Claude) are strictly prohibited.
* **Core Goal:** Overcome vocabulary mismatch, script fragmentation, and ungrounded generation by building a robust cross-lingual retrieval pipeline with verifiable evidence grounding and safe abstention.

### Functional Requirements Matrix
* **FR-1 (Multilingual Input):** Accepts queries in Urdu script, Roman Urdu, English, and code-mixed formats.
* **FR-2 (Query Normalization):** Normalizes Unicode inconsistencies, character elongation, spelling variations, and Eastern Arabic/Urdu digits.
* **FR-3 (Information Retrieval):** Implements and benchmarks at least two distinct approaches: Lexical (Okapi BM25) and Dense Semantic (MiniLM-L12), plus a Hybrid Reciprocal Rank Fusion (RRF) pipeline.
* **FR-4 (Evidence-Grounded Answers):** Every substantive answer is strictly backed by official retrieved passages and verified source URLs.
* **FR-5 (Safe Abstention):** Safely abstains on ambiguous questions, out-of-domain queries, personal guarantees, time-sensitive inquiries, and prompt injections.
* **FR-6 (Traceability):** Preserves document identifiers (`F01`-`F30`), passage excerpts, and official URLs for citizen auditability.
* **FR-7 (Evaluation):** Full quantitative benchmark evaluation on 150 official records (60 held-out test queries, 90 train/dev queries).
* **FR-8 (Usability):** Accessible Streamlit interface with language selection, 12 rubric validation cases, live latency metrics, and downloadable deliverables.

### Judging Rubric Alignment (100 Points Total)
| Criterion | Points | System Implementation & Evidence |
| :--- | :---: | :--- |
| **NLP Innovation & Technical Depth** | 20 | Dual-script Unicode cleaner, phonetic collapse lexicon, subword 3-grams, and Hybrid RRF fusion ($k=60$). |
| **Retrieval Quality & Relevance** | 20 | 92.3% Recall@5 (BM25) and 0.8150 MRR@5 on held-out test set; 90.8% Recall@5 on Roman Urdu queries. |
| **Answer Reliability & Grounding** | 20 | 97.6% Abstention F1, zero hallucination, strict passage citation, and verified official DGI&P URLs. |
| **Validation & Experimental Rigor** | 15 | Multi-engine comparison table, disaggregated language splits, 4 qualitative failure case studies, and 18 unit tests. |
| **Working Prototype & Usability** | 10 | Streamlit 4-tab dashboard, response language switcher, interactive 12-case rubric runner, compact responsive layout. |
| **Practical Impact & Relevance** | 10 | Directly addresses Pakistan public service accessibility across non-standard Roman Urdu and Urdu script. |
| **Presentation & Demonstration** | 5 | 5-page publication PDF technical report, structured validation pack ZIP, and clear documentation. |

---

## 💻 Complete Technology Stack

* **Programming Language:** Python 3.12 (compatible with Python 3.10+)
* **Information Retrieval (Lexical):** `rank-bm25` (Okapi BM25 algorithm, $k_1=1.5, b=0.75$) with language-specific stopword filtering and subword character 3-gram indexing.
* **Information Retrieval (Dense):** HuggingFace `sentence-transformers` (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 117M parameters, 384-dimensional embeddings, PyTorch CPU execution).
* **Rank Fusion Engine:** Reciprocal Rank Fusion (RRF with $k=60$) coupled with min-max normalized score combination (65% dense weight, 35% lexical weight).
* **Web Interface:** `streamlit` 1.66+ with custom responsive CSS, Nastaliq typography, session state caching, and mobile-friendly viewport optimization.
* **Document & PDF Generation:** `reportlab` 4.x (two-pass dynamic `NumberedCanvas` generating publication-quality 5-page PDF reports), `PyPDF2`.
* **Data Processing & Scientific Computing:** `numpy`, `pandas`, `unicodedata`, `re`.
* **Execution & Sovereign Edge Policy:** 100% offline local inference via `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Zero GPU requirement (CPU-optimized, average retrieval latency < 30 ms).

---

## 🚀 Key Innovations & System Architecture

1. **Dual-Script Preprocessing & Normalization (`src/normalizer.py`):**
   * **Urdu Script Unicode Cleaner:** Normalizes non-standard Alef (`آ, أ, إ` $\to$ `ا`), Yeh (`ي, ى, ے` $\to$ `ی / ے`), Heh (`ہ, ة` $\to$ `ہ`), and Kaf (`ك` $\to$ `ک`), strips diacritics / Tashkeel, and converts Eastern Arabic and Urdu digits (`۰-۹`) to standard numerals.
   * **Roman Urdu Phonetic Normalizer:** Vocabulary collapse for common auxiliary verbs (`hy/hei/he` $\to$ `hai`, `krna` $\to$ `karna`), domain terms (`pasport` $\to$ `passport`, `renu` $\to$ `renew`, `chahye/chaheay` $\to$ `chahiye`), and collapses character elongation (`paspooort` $\to$ `passport`).
   * **Subword 3-Gram Indexing:** Generates character n-grams to recover relevance even under unseen phonetic typos.

2. **Multi-Engine Retrieval Architecture (`src/retriever.py`):**
   * **BM25 Lexical:** Fast inverted-index lookup optimized for exact document numbers, fee amounts, and procedural keywords.
   * **Dense MiniLM Semantic:** Multilingual vector representations capturing conceptual intent across paraphrased questions.
   * **Hybrid RRF Fusion:** Merges lexical exactness with semantic generalizations, achieving 100.0% cross-script consistency.

3. **Safe Abstention & Guardrails (`src/generator.py`):**
   * **Ambiguity Interception:** Intercepts unspecific queries (e.g. *"Kitni fee hai?"*) and asks for required parameters (booklet size, validity, delivery tier).
   * **Out-of-Domain Detection:** Calibrated semantic threshold ($\tau = 0.38$) rejects non-public-service queries (sports, weather, politics).
   * **Adversarial & Injection Immunity:** Detects attempts to bypass citations or override instructions.
   * **Personal Guarantees Refusal:** Directs citizens to official tracking channels (SMS 9988) instead of guessing individual delivery dates.

---

## 📊 Quantitative Benchmark Results

Evaluated across **150 benchmark records** (60 held-out test queries, 90 train/dev queries):

### Primary Benchmark: Held-Out Test Set ($N=60$)
| Retrieval Method | Recall@1 | Recall@5 | MRR@5 | Citation Support | Abstention F1 | Consistency | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Lexical)** | 74.4% | **92.3%** | **0.8150** | 74.4% | **97.6%** | **100.0%** | **0.8 ms** |
| **Dense Semantic (MiniLM)** | 59.0% | 76.9% | 0.6603 | 56.4% | 97.7% | **100.0%** | 17.5 ms |
| **Hybrid (RRF Fusion)** | **76.9%** | 84.6% | 0.7957 | 71.8% | 95.5% | **100.0%** | 20.8 ms |

### Full Dataset ($N=150$) Disaggregation by Language
| Language / Script | Total Queries | BM25 Recall@5 | BM25 MRR@5 | Dense Recall@5 | Hybrid Recall@5 | Hybrid MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **English (EN)** | 33 queries | **93.9%** | 0.8586 | 87.9% | **93.9%** | **0.8712** |
| **Urdu Script (UR)** | 31 queries | **93.5%** | **0.8414** | 83.9% | 90.3% | 0.8333 |
| **Roman Urdu (ROMAN_UR)** | 65 queries | **90.8%** | **0.7967** | 72.3% | 83.1% | 0.7454 |

---

## 🧪 Verification of the 12 Rubric Validation Cases

| Case | Category | Sample Query | System Behavior & Evidence Citation | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Urdu Script | `پاسپورٹ کی تجدید کے لیے کون سی دستاویزات درکار ہیں؟` | Answers in Urdu; cites Passage F18 with DGI&P FAQ link. | **PASS** |
| **2** | Roman Urdu | `Passport renew karwanay ke liye kya documents chahiye?` | Identifies renewal intent; matches Case 1 evidence without translation. | **PASS** |
| **3** | English Query | `Which documents are required to renew a passport?` | Cites Passage F16/F18 with official procedural requirements. | **PASS** |
| **4** | Code-Mixed | `Passport renewal ke liye online apply karna hota hai ya office jana hota hai?` | Clarifies portal eligibility criteria (F17/F18) vs office visit requirements. | **PASS** |
| **5** | Spelling Variation | `pasport renu karwana hy` vs `passport renew karwana hai` | Normalizer and subword 3-grams recover F17 for both variations identically. | **PASS** |
| **6** | Paraphrased Query | `I need to replace my expired passport. Where should I start?` | Dense semantic model retrieves renewal guidelines (F16/F17). | **PASS** |
| **7** | Unsupported Personal | `What is the guaranteed processing time for my individual application?` | Safely abstains; clarifies general timelines (21 days) cannot guarantee individual delivery. | **PASS** |
| **8** | Out-of-Domain | `Who will win the next cricket World Cup?` | Cosine similarity = 0.114 < 0.38; clearly communicates question is outside public services. | **PASS** |
| **9** | Ambiguous Query | `How much does it cost?` / `Kitni fee hai?` | Asks for clarification on pages (36/72/100), validity (5/10 yr), and service tier. | **PASS** |
| **10** | Conflicting Evidence | `One page says 21 days for normal processing, but another says different.` | Discloses discrepancy, cites 21-day standard, and directs to current live portal advisory. | **PASS** |
| **11** | Citation Integrity | `What is required if my passport has been lost?` | Cites Passage F05 police report requirement with exact official URL. | **PASS** |
| **12** | Semantic Noise | `asdfghjkl zxcvbnm 12345` | Rejects random string and requests meaningful procedural question. | **PASS** |

---

## 🛠️ Step-by-Step Installation & Real Execution Commands

All commands below are actual, verified terminal commands designed for Windows PowerShell and standard Bash environments:

### 1. Prerequisites
* Python 3.10, 3.11, or 3.12 installed
* Git installed
* Standard CPU (No GPU required)

### 2. Clone the Repository
```bash
git clone https://github.com/AbdulAzeemHashmi/Multilingual-E-Sahulat-AI.git
cd Multilingual-E-Sahulat-AI
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Knowledge Base & Pre-Compute Embeddings
```bash
python src/data_loader.py
```
*Outputs: `data/knowledge_base.json`, `data/train_dev.json`, `data/test.json`, and `data/passage_embeddings.npz`.*

### 5. Run the Quantitative Evaluation Benchmark
```bash
python src/evaluate.py
```
*Outputs: Full retrieval metrics table, CSV comparisons, and held-out test predictions.*

### 6. Run the Automated Unit Test Suite
```bash
python -m unittest discover tests
```
*Runs all 18 automated unit tests covering normalization, retrieval modes, traceability, and abstention guardrails.*

### 7. Generate the 5-Page Publication Technical Report (PDF)
```bash
python src/generate_report.py
```
*Outputs: `reports/Multilingual_E_Sahulat_AI_Technical_Report.pdf`.*

### 8. Launch the Streamlit Web Application
Use the exact command for your environment:

* **Windows PowerShell (Recommended):**
  ```powershell
  python -m streamlit run app.py
  ```
* **Command Prompt / Terminal with Streamlit on PATH:**
  ```bash
  streamlit run app.py
  ```
* Open your browser and navigate to:
  ```
  http://localhost:8501
  ```

---

## 📂 Updated Repository Structure

```
Multilingual-E-Sahulat-AI/
├── app.py                                          # 4-Tab Streamlit Web Application (UI/UX Optimized)
├── requirements.txt                                # Production dependencies list
├── README.md                                       # Comprehensive Documentation & Tech Stack Guide
├── Multilingual E-Sahulat AI_Hackathon_Dataset.txt # Official 150-record DGI&P dataset
├── data/
│   ├── knowledge_base.json                         # 30 canonical official public-service procedures (F01-F30)
│   ├── train_dev.json                              # 90 train/dev queries with ground truth annotations
│   ├── test.json                                   # 60 held-out test queries
│   └── passage_embeddings.npz                      # Pre-computed 384-dimensional dense embedding cache
├── src/
│   ├── __init__.py                                 # Python package identifier
│   ├── normalizer.py                               # Urdu Unicode cleaner, Roman Urdu normalizer, subword 3-grams
│   ├── retriever.py                                # Okapi BM25, Dense MiniLM-L12, Hybrid RRF rank fusion
│   ├── generator.py                                # Grounded answer synthesis & 9-category safe abstention engine
│   ├── evaluate.py                                 # Full evaluation suite (Recall@K, MRR@K, Abstention F1)
│   ├── data_loader.py                              # Dataset parser & knowledge base builder
│   └── generate_report.py                          # ReportLab 5-page publication PDF report generator
├── tests/                                          # Automated Unit Test Suite (18 tests, 100% pass rate)
│   ├── test_normalizer.py                          # Script detection, Urdu Unicode cleaning, Roman Urdu collapse
│   ├── test_retriever.py                           # BM25, Dense, Hybrid retrieval, score thresholds, source URLs
│   └── test_generator.py                           # All 9 abstention guardrails and grounded citation generation
├── reports/
│   └── Multilingual_E_Sahulat_AI_Technical_Report.pdf  # Deliverable 1: 5-page PDF technical publication report
└── validation_pack/                                # Deliverables 2 & 3: Submission Package
    ├── Multilingual_E_Sahulat_AI_Validation_Pack.zip   # Complete standalone submission ZIP archive
    ├── retrieval_comparison.csv                     # Multi-engine comparison metrics (CSV)
    ├── retrieval_comparison.md                      # Multi-engine comparison metrics (Markdown)
    ├── test_predictions.csv                        # 60 held-out test predictions sheet
    ├── test_predictions.json                       # 60 test predictions with complete score metadata
    ├── full_evaluation_metrics.json                # Complete metrics across test set and full dataset
    └── failure_analysis.md                         # Qualitative failure analysis (4 representative case studies)
```

---

## 📜 Citations & Source Provenance

* **Official Source:** Directorate General of Immigration & Passports (DGI&P), Ministry of Interior, Government of Pakistan (`https://dgip.gov.pk`).
* **Underlying Pretrained Model:** HuggingFace `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0).
* **Lexical Model:** Okapi BM25 (`rank-bm25`).
* **License:** Open Source MIT License.
