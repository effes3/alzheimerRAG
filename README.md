<div align="center">

# AlzheimerRAG 🧬 Precision Biomedical Retrieval

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20Search-orange)]()
[![Built with](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)

**A high-precision search tool for biomedical literature**  
Solves the **specificity-recall trade-off** problem (e.g., distinguishing between *APOE3* and *APOE4*) using metadata-based re-ranking and a hybrid search architecture

[Demo](#-demo-quick-look) • [Quick start](#-quick-start) • [Architecture](#-retrieval-architecture) • [Results](#-performance-evaluation-results10)

</div>

---

## 🎬 Demo: Quick Look

![Application Demo](use_showcase.gif)  
*Query → Retrieved Papers → Entity-Aware Highlighting*

---

## 🚀 Quick Start

### 1. Prerequisites
*   **API Keys:** Access to OpenRouter (Llama-3/Grok/Qwen and etc. models) is required

### 2. Configuration
```bash
cp .env.example .env
```
Add your keys to `.env`:
```env
OPENROUTER_API_KEY="sk-or-v1-..."
RAG_MODEL_NAME="arcee-ai/trinity-mini:free"
```

### 3. Installation & Execution
I use `uv` for deterministic environment assembly

**Option A: via `uv` (Recommended)**
```bash
git clone https://github.com/effes3/alzheimerRAG.git
cd alzheimerRAG
uv sync
uv run python src/chromadb_builder.py
uv run streamlit run src/app.py
```

<details>
<summary><strong>Option B: via `pip` (Slow)</strong></summary>

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

## 📂 Project Structure
<details>
<summary>Show project tree</summary>

```bash
alzheimerRAG/
├── data/           # PDFs, extracted texts, and entity annotations
├── scripts/        # Data processing and EDA scripts
├── src/            # Main application and retriever logic
├── results/        # Reports and evaluation results
├── README.md
├── requirements.txt
└── pyproject.toml
```
</details>

---

## 🧬 Retrieval Architecture

Hybrid search combines **Semantic Density** and **Lexical Exactness**, enhanced by the **Metadata Injection** mechanism

### Scoring Logic
The formula for the final document score ($Score(d)$):

$$
Score(d) = \underbrace{\left[ \alpha \left(1 - \frac{S_{\text{vec}}}{Max_{\text{vec}}}\right) + (1-\alpha) \left(\frac{S_{\text{bm25}}}{Max_{\text{bm25}}}\right) \right]}_{\text{Hybrid Base Score}} \times \underbrace{\text{Boost}(Entities)}_{\text{Metadata Multiplier}}
$$

| Component | Implementation | Why is it needed? |
| :--- | :--- | :--- |
| **S_vec** 🟢 | `NeuML/pubmedbert` | Captures conceptual similarity ("meaning") |
| **S_bm25** 🔵 | **BM25Okapi** | Ensures exact term matching |
| **Boost** ⚡ | Metadata Injection | Multiplies the score if entities from the query are present in the metadata |

---

## 📊 Performance Evaluation (Results@10)

**Infrastructure:** Judge: `gpt-4o-mini` | Generator: `gemma-2-27b-it` | Query Expansion: `qwen3-4b`

| Architecture | Faithfulness | Relevancy | Precision | Recall | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vector + Entity Boost** 🏆 | **0.816** | **0.410** | **0.706** | **0.579** | **9.80** |
| Hybrid + Entity Boost | 0.800 | 0.367 | 0.706 | 0.579 | 11.25 |
| Hybrid + HyDE + Boost | 0.777 | 0.310 | 0.596 | 0.421 | 14.79 |

**Key insights:**
1. **Simplicity wins:** The Vector + Entity Boost configuration showed the best faithfulness with minimal latency
2. **HyDE is noisy:** Using hypothetical embeddings worsened metrics due to hallucinations in the biomedical context
3. **Recall Ceiling:** The identical recall ceiling indicates a bottleneck in the ingestion stage rather than retrieval

---

## 🔮 Future Roadmap
*   🔧 **Advanced Ingestion:** Implement Parent-Child Chunking to preserve the context of small fragments
*   🖼️ **Vision-Language Models:** Transition to VLM (e.g., ColPali) for extracting data from tables and graphs
*   🌐 **GraphRAG Integration:** Use knowledge graphs to map gene-pathology relationships
*   🧪 **Fine-tuned PubMedBERT:** Retrain the model on a narrow corpus of articles on Alzheimer's disease

---

## ⚠️ Limitations
<details>
<summary>Click to expand</summary>

*   **Dataset Constraints:** The evaluation was performed on a network of 24 papers. The behaviour may change on million-scale samples
*   **Table Parsing:** The current pipeline ignores statistical tables
*   **Semi-Automated Ingestion:** Entity extraction is currently implemented via the Grok web interface; a full transition to API is planned

</details>


