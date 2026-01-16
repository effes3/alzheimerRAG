import os
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig 
from langchain_openai import ChatOpenAI
from rag_agent import AlzheimerRAGAgent
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
import time
import numpy as np
load_dotenv()

current_dir = Path(__file__).resolve().parent
kb_path = current_dir.parent / "data" / "chromadb" / "with_llm"

JUDGE_MODEL = os.getenv("JUDGE_MODEL_NAME", "openai/gpt-4o-mini")
RAG_MODEL = os.getenv("RAG_MODEL_NAME", "google/gemma-3-27b-it:free")
KB_PATH = "./chromadb/with_llm"
print(f"⚖️ Evaluator Judge: {JUDGE_MODEL}")
print(f"🧪 RAG Subject: {RAG_MODEL}")

judge_llm = ChatOpenAI(
    model=JUDGE_MODEL, 
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)

judge_embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

current_dir = Path(__file__).parent
kb_path = current_dir / "chromadb" / "with_llm"

print("🤖 Initializing Agent...")
agent = AlzheimerRAGAgent(
    kb_path=str(kb_path),
    model_name=os.getenv("RAG_MODEL_NAME"),
    use_hybrid_search=True,
    alpha=0.7,
    k_results=3,
    use_hyde=True
)

eval_data = [
    {
        "question": "Which five proteins comprise the plasma proteomic signature that accurately predicts APOE ε4 carrier status?",
        "ground_truth": "The machine learning model uses five proteins—SPC25, NEFL, S100A13, TBCA, and LRRN1—to predict APOE ε4 carrier status with high accuracy."
    },
    {
        "question": "What mechanism does the Bassoon protein use to exacerbates tau pathology in Alzheimer’s disease?",
        "ground_truth": "The Bassoon protein interacts with high-molecular-weight tau aggregates, stabilizes tau seeds, and promotes tau propagation and neurotoxicity in the brain."
    },
    {
        "question": "What is the primary neuroprotective mechanism of the PILRA protein against Alzheimer's disease related to viral infection?",
        "ground_truth": "PILRA may confer protection by limiting HSV-1 viral entry through reduced interaction with viral glycoprotein D, thereby attenuating neuroinflammation and Aβ accumulation."
    },
    {
        "question": "How do Siglec-7 and Siglec-9 contribute to the risk of Alzheimer's disease according to mediation analysis?",
        "ground_truth": "Siglec-7 and Siglec-9 increase the risk of Alzheimer's disease through p-tau in the cerebrospinal fluid (CSF), highlighting tau-driven neuroinflammation as a pathogenic mechanism."
    },
    {
        "question": "What specific cholesterol-related defect was identified in APOE4-carrying oligodendrocytes during the Brain Aging Symposium?",
        "ground_truth": "APOE4 carriers showed reduced myelination due to impaired cholesterol transport in the endoplasmic reticulum, representing a cholesterol biosynthesis defect in oligodendrocytes."
    },
    {
        "question": "What is the proposed therapeutic action of O-GlcNAcase (OGA) inhibitors in tau pathology?",
        "ground_truth": "OGA inhibitors increase the O-GlcNAcylation of tau, which stabilizes the protein and reduces its aggregation and hyperphosphorylation."
    },
    {
        "question": "Which bioactive compound from Melissa officinalis oil showed the highest affinity for acetylcholinesterase, neprilysin, and γ-Secretase in docking studies?",
        "ground_truth": "Sagerinic acid (MO-32) exhibited the highest affinity against neprilysin, γ-Secretase, and acetylcholinesterase with significant docking scores."
    },
    {
        "question": "How does quercetagitrin modulate Alzheimer-like pathologies in the APP/PS1 mouse model?",
        "ground_truth": "Quercetagitrin reduces the levels of TNFα and inhibits the activation of NFκB signaling, thereby ameliorating Aβ pathology, cognitive impairments, and neuroinflammation."
    },
    {
        "question": "What is the role of ASC specks in the propagation of Alzheimer's disease pathology?",
        "ground_truth": "Inflammasome-activated microglia release ASC specks that bind to Aβ, promoting the aggregation and spread of amyloid-beta plaques."
    },
    {
        "question": "Which target combination is used in the drug 'donecopride' to treat Alzheimer’s symptoms?",
        "ground_truth": "Donecopride is a hybrid molecule that acts as both an acetylcholinesterase (AChE) inhibitor and a 5-HT4 receptor agonist."
    },
    {
        "question": "How do DEPTAC and PhosTAC molecules function to reduce tau pathology?",
        "ground_truth": "Both dephosphorylation targeting chimeras (DEPTAC) and phosphorylation targeting chimeras (PhosTAC) recruit Protein Phosphatase 2A (PP2A) to dephosphorylate tau, reducing its aggregation and neurotoxicity."
    },
    {
        "question": "Which proteins consistently show decreased abundance in Alzheimer’s disease plasma across multiple cohorts according to GNPC analysis?",
        "ground_truth": "GNPC meta-analysis identified 130 proteins consistently lower in AD plasma, including VAT1, GPD1, ARPC2, PA2G4, RPS12, NPTXR, and NT5C."
    },
    {
        "question": "What is the effect of the TREM2-APOE pathway on microglial phenotypes in neurodegenerative diseases?",
        "ground_truth": "The TREM2-APOE pathway drives the transcriptional phenotypic shift of microglia from a homeostatic state to a neurodegenerative (DAM) phenotype after phagocytosis of apoptotic neurons."
    },
    {
        "question": "What specific mechanism does resveratrol use to clear Aβ in Alzheimer's disease models?",
        "ground_truth": "Resveratrol facilitates Aβ clearance by upregulating the expression of neprilysin and activating autophagy."
    },
    {
        "question": "How does Siglec-3 (CD33) impact amyloid-beta accumulation in the brain?",
        "ground_truth": "Siglec-3 promotes pro-inflammatory microglial activation, which potentially accelerates Aβ accumulation and impairs microglial phagocytosis."
    },
    {
        "question": "What is the role of C1q in early synapse loss in Alzheimer's mouse models?",
        "ground_truth": "C1q attaches to synapses before the formation of plaques and triggers microglial CR3-mediated phagocytosis, leading to aberrant complement-driven synaptic pruning."
    },
    {
        "question": "Which Chinese traditional medication is currently in Phase 2 trials to enhance brain circulation and increase Aβ degradation in the liver?",
        "ground_truth": "Yangxue Qingnao is being assessed in a Phase 2 study (NCT04780399) for its ability to enhance circulation and liver-mediated Aβ degradation."
    },
    {
        "question": "How do PROTACs (Proteolysis-Targeting Chimeras) facilitate the degradation of pathogenic proteins in AD?",
        "ground_truth": "PROTACs are heterobifunctional molecules that recruit target proteins (like tau) to E3 ligases, marking them for degradation by the ubiquitin-proteasome system."
    },
    {
        "question": "What is the therapeutic benefit of inhibiting O-GlcNAcase (OGA) in the brain of AD patients?",
        "ground_truth": "Inhibiting OGA reduces the formation of pathological tau and ameliorates neurodegeneration by increasing the O-GlcNAcylation of tau protein."
    }
]

questions = [item["question"] for item in eval_data]
ground_truths = [item["ground_truth"] for item in eval_data]
answers = []
contexts = []
latencies = []

print(f"🧪 Generating answers for {len(questions)} questions...")

for q in questions:
    print(f"   Asking: {q}")
    start_time = time.time()
    result = agent.query(q, return_sources=True)
    end_time = time.time()
    latency = end_time - start_time
    latencies.append(latency)
    answers.append(result["answer"])
    contexts.append([s["full_text"] for s in result["sources"]])

print("\n📊 Running Ragas metrics with RPM Guard (Step-by-step)... ")

final_results_list = []
output_file = "rag_evaluation_partial.csv"

safe_config = RunConfig(
    max_workers=1, 
    timeout=300,    
    max_retries=3
)

metrics_to_use = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

for m in metrics_to_use:
    if hasattr(m, "llm"):
        m.llm = judge_llm
    if hasattr(m, "embeddings"): 
        m.embeddings = judge_embeddings

for i in range(len(questions)):
    print(f"🔄 Evaluating row {i+1}/{len(questions)}...")

    single_row_dict = {
        "question": [questions[i]],
        "answer": [answers[i]],
        "contexts": [contexts[i]],
        "ground_truth": [ground_truths[i]]
    }
    single_dataset = Dataset.from_dict(single_row_dict)

    try:
        row_result = evaluate(
            dataset=single_dataset,
            metrics=metrics_to_use,
            run_config=safe_config
        )
        
        res_df = row_result.to_pandas()
        res_df['latency_seconds'] = latencies[i]
        final_results_list.append(res_df)
        
        pd.concat(final_results_list).to_csv(output_file, index=False)
        
    except Exception as e:
        print(f"⚠️ Error on row {i+1}: {e}")
        continue

    if i < len(questions) - 1:
        wait_time = 1
        print(f"⏳ Sleeping for {wait_time}s to avoid RateLimit...")
        time.sleep(wait_time)
        
print("📊 Running Ragas metrics via OpenRouter...")

avg_latency = np.mean(latencies)

final_df = pd.concat(final_results_list)
print("\n" + "="*50)
print("🏆 FINAL AVERAGES")
print("="*50)
print(final_df[metrics_to_use].mean())
print(f"Average Latency: {np.mean(latencies):.2f}s")