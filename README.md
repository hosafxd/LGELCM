# LGELCM
# LLM-Guided Entity-Level Clinical Metric 
**License:** Pending academic paper publication

**Status:** In Development | **Target:** Academic Publication  
**Domain:** Medical Natural Language Processing | **Focus:** Radiology Report Entity Extraction

---

## Overview of comparison part of the project

A comprehensive evaluation system for assessing the quality of **automated medical entity extraction** from radiology reports. This framework addresses a critical challenge in medical NLP: comparing machine-generated schemas against ground truth annotations when sentence boundaries differ.

**Core Problem:** Traditional evaluation methods fail when comparing schemas with identical content but different text segmentation. Our framework solves this by implementing entity-level matching regardless of input splitting strategies.

---

## Problem Statement

Medical NLP models extract structured information from unstructured radiology reports. However, evaluating these models is challenging when:

- Reports are identical but split into different numbers of sentences
- Entities are merged or split across predictions
- Semantic similarity doesn't correlate with structural accuracy
- Clinical equivalence differs from exact string matching

**Example:**
```
Ground Truth:    20 sentences → 45 entities
Model Prediction: 18 sentences → 43 entities (2 merged)
Traditional metrics: Fail to align
Our framework: Entity-level matching with semantic similarity
```

---

## Key Features

### Multi-Level Evaluation Pipeline

**1. Structural Evaluation**
- Entity-level precision, recall, and F1-score
- Field-weighted scoring based on clinical importance
- Detection of merged/split entities and contradictions
- Automatic schema format detection (handles legacy formats)

**2. Semantic Analysis**
- Medical domain embeddings (PubMedBERT, Clinical-BERT variants)
- Weighted schema-to-text conversion emphasizing critical fields
- Cosine similarity across multiple embedding models
- SapBERT-based semantic entity matching

**3. LLM-Based Clinical Validation**
- Gemini 2.5 Pro for clinical equivalence assessment
- Retry logic with exponential backoff for API reliability
- Safety configurations to handle medical content
- Clinical impact severity classification

**4. Error Taxonomy**
- Merged entity detection (e.g., combining separate findings)
- Split entity identification (over-granular extraction)
- Presence contradiction tracking (present ↔ absent mismatches)
- Degree incompatibility detection (conflicting severity modifiers)

---

## Technical Stack

### Core Technologies
- **Python 3.9+** - Primary language
- **PyTorch** - Deep learning backend
- **Sentence Transformers** - Embedding computation
- **Google Gemini API** - LLM evaluation
- **NumPy/SciPy** - Statistical analysis

### Medical NLP Models
- **PubMedBERT** - Biomedical text understanding
- **S-PubMedBERT** - Sentence-level medical embeddings
- **SapBERT** - Synonym-aware medical entity linking
- **Clinical-BERT** - Clinical note processing

### Evaluation Components
- Entity-level semantic matching
- Field-wise accuracy computation
- Statistical significance testing (bootstrap CI, paired t-tests)
- Multi-model comparison with aggregated metrics

---

## Methodology

### Entity Matching Strategy

```
1. Flatten all entities from all text segments
2. Encode with medical domain embeddings (SapBERT)
3. Compute pairwise semantic similarities
4. Apply optimal assignment via threshold matching (>0.6)
5. Classify unmatched entities as false positives/negatives
```

### Field Importance Weighting

```
finding_presence: 30%  (Most critical - present/absent/uncertain)
general_finding:  20%  (Primary diagnosis)
specific_finding: 20%  (Detailed diagnosis)
location:         15%  (Anatomical region)
degree:           10%  (Severity modifier)
measurement:       3%  (Quantitative values)
comparison:        2%  (Temporal changes)
```

### Evaluation Metrics

**Structural Metrics:**
- Precision, Recall, F1-score at entity level
- Field-wise accuracy across all schema fields
- Weighted overall score based on clinical importance

**Semantic Metrics:**
- Cosine similarity across 4 medical embedding models
- Average match quality for aligned entities
- Model-specific semantic scores

**Clinical Metrics:**
- LLM-assessed clinical equivalence (0.0-1.0)
- Critical error identification
- Clinical impact severity (critical/moderate/minor)

---

## Project Structure

```
medical-schema-evaluation/
├── entity_level_evaluator.py      # Core entity matching logic
├── medical_schema_evaluator.py    # Structural comparison
├── multi_embedding_evaluator.py   # Semantic similarity computation
├── llm_evaluator.py                # LLM-based validation
├── comprehensive_evaluation.py     # Main evaluation pipeline
└── data_report/                    # Evaluation datasets
```

---

## Use Cases

### Model Development
- Fine-tuning medical NER models with accurate performance metrics
- Comparing different entity extraction architectures
- Identifying systematic errors (merging, splitting, contradictions)

### Quality Assurance
- Validating automated annotation pipelines
- Detecting annotation inconsistencies
- Benchmarking human annotator agreement

### Research Applications
- Evaluating RAG systems for medical question answering
- Assessing LLM performance on structured medical data extraction
- Comparing rule-based vs. neural entity extraction approaches

---

## Current Status

**Implemented:**
- Entity-level semantic matching with SapBERT
- Multi-embedding evaluation (4 medical models)
- Structural error detection (merged/split entities)
- LLM integration with Gemini 2.5
- Standardized output format with comprehensive metrics
- Statistical analysis toolkit

**In Progress:**
- Cross-dataset validation across multiple radiology report types
- Statistical significance testing with bootstrap confidence intervals
- Performance optimization for large-scale evaluation

**Planned:**
- Academic publication with comparative analysis
- Extended support for additional medical document types
- Integration with common medical NLP benchmarks

---

## Performance Considerations

**Semantic Matching:** SapBERT encoding is compute-intensive. For large datasets:
- Consider GPU acceleration (10x speedup)
- Implement embedding caching
- Batch processing for efficiency

**LLM Evaluation:** Rate limits apply to free-tier APIs:
- Gemini Free Tier: 2M tokens/day
- Built-in retry logic with exponential backoff
- Configurable sleep intervals between requests

---

## Applications in Medical NLP

This framework supports evaluation of:
- **Named Entity Recognition (NER)** - Extracting medical entities from free text
- **Relation Extraction** - Identifying relationships between entities
- **Schema Mapping** - Converting unstructured to structured medical data
- **Information Retrieval** - Assessing retrieval quality in medical QA systems
- **Annotation Quality Control** - Validating human annotation consistency

---

## Why This Matters

Traditional NLP evaluation metrics (BLEU, ROUGE, exact match) fail in medical contexts where:
- Clinical meaning matters more than exact phrasing
- Sentence segmentation varies by annotator/model
- Semantic similarity doesn't guarantee clinical equivalence
- Entity relationships affect interpretation

This framework bridges the gap between **technical accuracy** and **clinical utility**.

---

## Future Work

- Publication in medical informatics conference/journal
- Expansion to multi-modal medical data (images + text)
- Integration with active learning pipelines
- Support for multilingual medical entity extraction
- Real-time evaluation API for production systems

---

## Technical Requirements

```
Python >= 3.9
sentence-transformers >= 2.2.0
torch >= 2.0.0
numpy >= 1.24.0
google-generativeai >= 0.3.0
scikit-learn >= 1.3.0
```

---

## Contact

For questions about methodology or collaboration opportunities, please open an issue or reach out via GitHub.

**Note:** This is an active research project. Code and documentation are being continuously refined. Star/watch the repository for updates.

---

