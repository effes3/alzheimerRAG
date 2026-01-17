# AlzheimerRAG: Precision Biomedical Retrieval

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Search-orange)]()
[![Built with](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)

A domain-specific RAG pipeline optimized for high-precision biomedical literature retrieval. Unlike standard semantic search, this system solves the **specificity-recall trade-off** (e.g., distinguishing *APOE3* from *APOE4*) by implementing metadata-aware re-ranking and a weighted hybrid search architecture

### Usage Showcase
![Application Demo](use_showcase.gif)

---

## 🚀 Quick Start

### Prerequisites
*   **API Keys:** This project uses OpenRouter to access LLMs

### Configuration
1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Add your API key to `.env`:
    ```bash
    OPENROUTER_API_KEY="sk-or-v1-..."
    RAG_MODEL_NAME="arcee-ai/trinity-mini:free"
    ```

### Installation & Execution
We use `uv` for deterministic builds

**Option A: uv (Recommended)**
```bash
# 1. Clone & Sync
git clone https://github.com/effes3/alzheimerRAG.git
cd alzheimerRAG
uv sync

# 2. Build Vector Index (Extracts entities & embeds text)
uv run python src/chromadb_builder.py

# 3. Launch UI
uv run streamlit run src/app.py
```

<details>
<summary><strong>Option B: pip (Legacy/Standard)</strong></summary>

```bash
git clone https://github.com/effes3/alzheimerRAG.git
cd alzheimerRAG

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

python src/chromadb_builder.py
streamlit run src/app.py
```
</details>

---

## 📂 Project tree (with no DB initialized yet)

```bash
alzheimerRAG/
├── data/
│   ├── alzheimer_papers/
│   │   ├── metadata.json (metadata of each article)
│   │   └── pmcids.txt # PMCID of each article (used for script that parses PDFs from PubMed)
│   └── data_processing/
│       ├── entities/
│       │   └── [25 files: *.json] (.json-files of PDFs where are entities for each article)
│       ├── merged/
│       │   ├── no_llm/
│       │   │   └── [24 files: *.json] (.json-files of PDFs with no using LLM as text-cleaner)
│       │   └── with_llm/
│       │       └── [24 files: *.json] (.json-files of PDFs with using LLM as text-cleaner)
│       ├── pdfs/
│       │   └── [24 files: *.pdf] (parsed pdfs from PubMed OA)
│       ├── pdfs_clean/
│       │   └── [24 files: *.pdf] (parsed pdfs from PubMed OA without article and references)
│       └── texts/
│           ├── texts_clean_no_llm/
│           │   └── [24 files: *.json] (temp files to make merged/)
│           └── texts_clean_w_llm/
│               └── [24 files: *.json] (temp files to make merged/)
├── pyproject.toml
├── README.md
├── requirements.txt
├── results/
│   ├── rag_evaluation_hsearch_funcscore.csv
│   ├── rag_evaluation_hsearch_funcscore_hyde.csv
│   └── rag_evaluation_vsearch_funcscore.csv
├── scripts/
│   ├── dataprep.py (make metadata collection)
│   ├── eda.ipynb (eda pipeline)
│   ├── ext.py (extract PubMed CIDs from metadata)
│   ├── merge_text_w_entities.py (merge text with entities from the semi-automated extraction pipeline)
│   ├── pdf2text.py (extract text from downloaded PDFs)
│   └── selen.py (script to download PDFs)
└── src/
    ├── app.py
    ├── chromadb_builder.py
    └── evaluate.py
    └── rag_agent.py
```
  
---

## 🧬 Retrieval Architecture

Standard dense retrievers often fail to distinguish between semantically similar but functionally distinct entities. This pipeline implements a **Hybrid Search** strategy with a custom scoring formula

### The Scoring Logic
Relevance ($$Score(d)$$) is a weighted linear combination of **Semantic Density** ($$S_{vec}$$) and **Lexical Exactness** ($$S_{bm25}$$), amplified by a **Metadata Boost**

$$
Score(d) = \underbrace{\left[ \alpha \left(1 - \frac{S_{vec}}{Max_{vec}}\right) + (1-\alpha) \left(\frac{S_{bm25}}{Max_{bm25}}\right) \right]}_{\text{Hybrid Base Score}} \times \underbrace{\text{Boost}(Entities)}_{\text{Metadata Multiplier}}
$$

| Component | Implementation | Why it matters |
| :--- | :--- | :--- |
| **$S_{vec}$** | `NeuML/pubmedbert` | Captures conceptual alignment (the "gist") |
| **$S_{bm25}$** | **BM25Okapi** | Enforces exact term matching (the "nomenclature") |
| **Boost** | Metadata Injection | Multiplies score if query entities (Genes, Proteins) match doc metadata |

### Entity-Aware Re-ranking
1.  **Extraction:** Grok extracts key entities (e.g., "Amyloid-beta") during ingestion
2.  **Injection:** Entities are serialized as metadata in ChromaDB
3.  **Boost:** Queries containing these entities trigger the boost, prioritizing "deep dive" documents over general mentions

---

## 📊 Performance Evaluation

Benchmarks were conducted using the **Ragas** framework against a synthetic test set generated by **Google NotebookLM**

*   **Infrastructure:** Judge: `gpt-4o-mini` | Generator: `gemma-2-27b-it` | Query Expansion: `qwen3-4b`

### Results@10

| Architecture | Faithfulness | Relevancy | Precision | Recall | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vector + Entity Boost** 🏆 | **0.816** | **0.410** | **0.706** | **0.579** | **9.80** |
| Hybrid + Entity Boost | 0.800 | 0.367 | **0.706** | **0.579** | 11.25 |
| Hybrid + HyDE + Boost | 0.777 | 0.310 | 0.596 | 0.421 | 14.79 |

### Key Findings
1.  **Simplicity Wins:** The **Vector + Entity Boost** configuration outperformed complex hybrid setups, delivering the highest Faithfulness (0.816) with the lowest latency
2.  **HyDE is Noisy:** Hypothetical Document Embeddings (HyDE) actively degraded performance by hallucinating biomedical context not present in the source text
3.  **The "Recall Ceiling":** The shared Recall cap (0.579) across models points to an ingestion bottleneck rather than a retrieval failure

---

## 🔮 Future Roadmap

Based on the evaluation benchmarks, the next phase focuses on breaking the "Recall Ceiling" and improving noise filtration:

*   **Advanced Ingestion Strategy:** Implementation of **Parent-Child Chunking** to retain the context of small snippets, addressing the ingestion bottleneck
*   **Vision-Language Models (VLM):** Replacing standard PDF parsing with VLM-based extraction (e.g., ColPali) to better capture data from biomedical tables and charts, which are currently lost
*   **GraphRAG Integration:** Moving beyond vector similarity to a Knowledge Graph approach to better map relationships between genes (APOE4) and pathologies (Amyloid plaques)
*   **Fine-tuned Embedding Model:** Fine-tuning `PubMedBERT` specifically on Alzheimer's pathology abstracts to improve domain separation

---

## ⚠️ Limitations & Disclaimer

*   **Dataset Constraints:** The current evaluation uses a closed set of 24 open-access PMC papers. Performance metrics may differ significantly on a chaotic, million-scale dataset
*   **Table Parsing:** The current text extraction pipeline strips out statistical tables, which often contain the core findings of clinical trials
*   **Semi-Automated Ingestion:** To optimize costs during the development phase, the Entity Extraction step currently relies on manual batch processing via the Grok web interface. In a production environment, this would be replaced by a programmatic API call (e.g., OpenAI/Anthropic) or a fine-tuned local NER model to ensure fully automated, end-to-end reproducibility. For a detailed breakdown of the prompting strategy and extraction workflow used, please refer to eda.ipynb



