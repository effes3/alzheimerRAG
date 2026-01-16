import os
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from chromadb_builder import AlzheimerKnowledgeBase

from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

def get_env_bool(key: str, default: bool) -> bool:
    """Helper to parse boolean strings from .env"""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 't', 'yes')

class AlzheimerRAGAgent:
    """
    RAG Agent for Alzheimer's Drug Target Discovery.
    Configured dynamically via .env or constructor arguments.
    """
    
    def __init__(
        self,
        kb_path: str = "./chromadb/with_llm",
        model_name: str = None,
        use_hybrid_search: bool = None,
        k_results: int = None,
        alpha: float = None,
        use_hyde: bool = None
    ):
        self.kb_path = Path(kb_path)
        
        # --- 1. CONFIGURATION LOADING ---
        # Logic: Arg -> Env -> Default
        
        self.model_name = model_name or os.getenv('RAG_MODEL_NAME', 'google/gemma-3-27b-it:free')
        self.hyde_model_name = os.getenv('HYDE_MODEL', 'qwen/qwen3-4b:free')
        
        self.k_results = k_results if k_results is not None else int(os.getenv('RETRIEVAL_K', 5))
        self.alpha = alpha if alpha is not None else float(os.getenv('RETRIEVAL_ALPHA', 0.7))
        
        # Booleans
        env_hybrid = get_env_bool('USE_HYBRID_SEARCH', True)
        self.use_hybrid_search = use_hybrid_search if use_hybrid_search is not None else env_hybrid
        
        env_hyde = get_env_bool('USE_HYDE', True)
        self.use_hyde = use_hyde if use_hyde is not None else env_hyde

        # Rate Limiting State
        self.request_count = 0
        self.last_reset_time = time.time()

        print(f"🤖 Agent Config: Model={self.model_name}, HyDE={self.hyde_model_name}")
        print(f"⚙️ Search Params: K={self.k_results}, Alpha={self.alpha}, Hybrid={self.use_hybrid_search}, HyDE={self.use_hyde}")

        # --- 2. KNOWLEDGE BASE ---
        print("📚 Loading knowledge base...")
        self.kb = AlzheimerKnowledgeBase(
            collection_name="alzheimer_papers",
            persist_directory=str(self.kb_path)
        )
        
        # Initialize Vectorstore and BM25
        vectorstore = self.kb.create_vectorstore(documents=[], reset=False)
        self.kb.vectorstore = vectorstore
        self.kb.load_bm25()

        if vectorstore is None:
            raise ValueError(f"CRITICAL: Could not load vectorstore from {self.kb_path}")
        
        # --- 3. LLM SETUP ---
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2000
        )

        self.fast_llm = ChatOpenAI(
            model=self.hyde_model_name,
            openai_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_base='https://openrouter.ai/api/v1',
            temperature=0.2,
            max_tokens=1000
        )
        
        self._build_chains()
        print("✅ RAG Agent initialized")

    def _build_chains(self):
        """Constructs LangChain pipelines"""
        
        # 1. Entity Extraction
        self.entity_prompt = ChatPromptTemplate.from_template(
            """Identify key biological entities (Proteins, Genes, Drugs, Receptors) in the question.
            Return ONLY a comma-separated list. If none, return nothing.
            Question: {question}
            Entities:"""
        )
        self.entity_chain = self.entity_prompt | self.fast_llm | StrOutputParser()

        # 2. HyDE
        self.hyde_prompt = ChatPromptTemplate.from_template(
            """You are an expert biomedical researcher. 
            Write a brief, scientific abstract (3-4 sentences) that would perfectly answer the following question about Alzheimer's disease.
            Do NOT answer the question directly, but write a hypothetical text passage that contains the answer.
            Use scientific terminology relevant to the field.
            Question: {question}
            Hypothetical Abstract:"""
        )
        self.hyde_chain = self.hyde_prompt | self.fast_llm | StrOutputParser()

        # 3. Main RAG
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert biomedical researcher specializing in Alzheimer's disease drug target discovery. 
            
            YOUR RULES:
            1. Strict alignment: If the question is NOT related to Alzheimer's/Biology/Pharma, refuse to answer and say: "I am a specialized assistant for Alzheimer's research and can only answer questions related to this domain."
            2. Base your answer ONLY on the provided context.
            3. Cite sources inline using [ArticleID, ChunkX].
            4. If the context is missing information, admit it explicitly.
            
            CONTEXT:
            {context}
            
            ENTITIES FOUND:
            {entities}
            """),
            ("human", "{question}"),
        ])
        
        self.chain = (
            {
                "context": RunnablePassthrough(),
                "entities": RunnablePassthrough(),
                "question": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _extract_query_entities(self, question: str) -> List[str]:
        try:
            result = self.entity_chain.invoke({"question": question})
            if not result or "none" in result.lower():
                return []
            entities = [e.strip() for e in result.split(',') if e.strip()]
            print(f"🧬 Entities: {entities}")
            return entities
        except Exception as e:
            print(f"⚠️ Entity extraction warning: {e}")
            return []

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            meta = doc.metadata
            formatted.append(
                f"[Source: {meta.get('article_id', 'Unknown')}, Chunk_{meta.get('chunk_index', 0)}]\n"
                f"Text: {doc.page_content}\n"
            )
        return "\n---\n".join(formatted)

    def _extract_entities_from_docs(self, docs: List[Document]) -> str:
        all_entities = set()
        for doc in docs:
            entities_str = doc.metadata.get('entities', '')
            if entities_str:
                all_entities.update([e.strip() for e in entities_str.split(',')])
        return ', '.join(sorted(all_entities))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """Executes the query pipeline with retries."""
        
        print(f"\n❓ Query: {question}")

        # 1. Analyze
        query_entities = self._extract_query_entities(question)
        search_query = question

        # 2. HyDE Expansion
        if self.use_hyde:
            hypothetical_doc = self.hyde_chain.invoke(question)
            print(f"🧠 HyDE Abstract (preview): {hypothetical_doc[:80]}...")
            search_query = hypothetical_doc

        # 3. Retrieve
        # Note: We pass original entities to hybrid_search for metadata boosting
        docs = self.kb.hybrid_search(
            query=search_query, 
            k=self.k_results, 
            alpha=self.alpha,
            filter_entities=query_entities
        )

        # 4. Generate
        context_str = self._format_docs(docs)
        entities_str = self._extract_entities_from_docs(docs)

        answer = self.chain.invoke({
            "question": question,
            "context": context_str,
            "entities": entities_str
        })

        if not return_sources:
            return {"answer": answer}

        sources = [{
            'article_id': d.metadata.get('article_id', 'Unknown'),
            'chunk_index': d.metadata.get('chunk_index', 0),
            'preview': d.page_content[:150] + '...',
            'full_text': d.page_content,
            'entities': d.metadata.get('entities', '')
        } for d in docs]

        return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    agent = AlzheimerRAGAgent()
    res = agent.query("What are the main protein targets for Alzheimer's disease treatment?")
    print("\n" + "="*50)
    print(res['answer'])