# ALZHEIMER RESEARCH RAG

Domain-specific retrieval-augmented generation (RAG) pipeline optimized for biomedical literature. The system prioritizes precision regarding specific biological entities (proteins, genes) through a hybrid search architecture and metadata-aware re-ranking

## Usage showcase

![](https://github.com/effes3/alzheimerRAG/use_showcase.gif)

## Configuration

Runtime behavior is controlled via environment variables

1.  Copy `.env.example` to `.env`
2.  Populate the required keys

```bash
OPENROUTER_API_KEY="sk-or-v1-..."
RAG_MODEL_NAME="arcee-ai/trinity-mini:free"
```

## Installation & Run

Dependency resolution utilizes `uv` for deterministic builds and optimal performance

### Option A: uv (Primary)

```bash
# Initialize
git clone https://github.com/effes3/alzheimer_rag.git
cd alzheimer_rag

# Sync dependencies and environment
uv sync

# Indexing
uv run python src/chromadb_builder.py

# Execution
uv run streamlit run src/app.py
```

### Option B: pip (Legacy)

```bash
# Setup
git clone https://github.com/effes3/alzheimer_rag.git
cd alzheimer_rag

# Environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Installation
pip install -r requirements.txt

# Indexing & Execution
python src/chromadb_builder.py
streamlit run src/app.py
```

## Retrieval Architecture

The core pipeline implements a **Hybrid Search** strategy to resolve the specificity-recall trade-off inherent in biomedical embedding models. Standard dense retrievers often fail to distinguish between semantically similar but functionally distinct entities (e.g., *APOE3* vs. *APOE4*)

### Scoring Logic

Document relevance ($d$) is computed via a weighted linear combination of dense vector similarity and sparse keyword matching, amplified by entity presence

$$
Score(d) = \left[ \alpha \left(1 - \frac{S_{vec}}{Max_{vec}}\right) + (1-\alpha) \left(\frac{S_{bm25}}{Max_{bm25}}\right) \right] \times \text{Boost}(Entities)
$$

| Component | Implementation | Function |
| :--- | :--- | :--- |
| **$S_{vec}$** | `NeuML/pubmedbert-base-embeddings` | Captures conceptual semantic alignment. |
| **$S_{bm25}$** | **BM25Okapi** | Enforces exact term matching for specific nomenclature. |
| **$\alpha$** | Float (Default: 0.7) | Tuning parameter balancing semantic vs. lexical weight. |
| **Boost** | Metadata Multiplier | Amplifies scores when query entities match document metadata. |

### Entity-Aware Re-ranking

1.  **Extraction:** Key biological entities (genes, treatments, proteins) are extracted from raw text via the Grok interface during ingestion
2.  **Injection:** Entities are serialized as metadata within the vector store
3.  **Execution:** Queries containing known entities trigger the boost mechanism, prioritizing documents explicitly analyzing the target subject over general references

Refer to `eda.ipynb` for distribution analysis of extracted entities

## Performance Metrics

Evaluation was conducted using the **Ragas** framework on a synthetic test set derived from domain literature

**infrastructure:**
*   **Judge:** `gpt-4o-mini`
*   **Generator:** `google/gemma-2-27b-it`
*   **Query Expansion:** `qwen/qwen3-4b`

**Results@10:**

| Architecture | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vector Search + Entity Boost** | **0.816** | **0.410** | **0.706** | **0.579** | **9.80** |
| Hybrid Search + Entity Boost | 0.800 | 0.367 | **0.706** | **0.579** | 11.25 |
| Hybrid + HyDE + Entity Boost | 0.777 | 0.310 | 0.596 | 0.421 | 14.79 |

### Analysis

1.  **Vector Search + Entity Boost** is the optimal configuration, delivering superior Faithfulness (0.816) and Relevancy (0.410) with the lowest latency
2.  **HyDE (Hypothetical Document Embeddings)** introduced significant noise. The query expansion mechanism tended to hallucinate biomedical context, degrading both precision and latency (+3.54s)
3.  **Hybrid Search** provided no statistical advantage over pure vector search when entity boosting was active, indicating the boosting mechanism successfully proxies the specificity benefits usually sought via BM25

### Evaluation Dataset

Ground truth generation utilized **Google NotebookLM** (Gemini 1.5 Pro). The model was constrained to generate Q&A pairs strictly from uploaded PDF contexts to minimize external knowledge contamination

## Known Constraints

**Ingestion Bottleneck:**
Context Recall (0.58) identifies the extraction layer as the primary point of failure. Complex PDF layouts (multi-column, floating figures) result in fragmented text chunks. This structural noise degrades embedding quality regardless of the retrieval strategy.

