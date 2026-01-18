import json
from pathlib import Path
from typing import List
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
import pickle
import os
import shutil

class AlzheimerKnowledgeBase:
    def __init__(
        self,
        collection_name: str = "alzheimer_papers",
        persist_directory: str = "data/chromadb",
        embedding_model: str = None
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True, parents=True)
        
        model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 500))

        print(f"🏗️ Builder Config: Model={model_name}, Chunk={self.chunk_size}, Overlap={self.chunk_overlap}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.bm25 = None
        self.bm25_docs = []
        self.bm25_metadata = []
        
        if (self.persist_directory / "chroma.sqlite3").exists():
            print(f"🔄 Found existing DB in {self.persist_directory}, loading...")
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
        else:
            self.vectorstore = None

        self.load_bm25()
    
    def build_from_markdown_with_entities(self, md_folder_path: str, entities_folder_path: str):
            md_folder = Path(md_folder_path)
            entities_folder = Path(entities_folder_path)
            
            if not md_folder.exists():
                raise FileNotFoundError(f"Directory with MD not found: {md_folder_path}")
            
            md_files = list(md_folder.glob("*.md"))
            print(f"\n📚 Downloading {len(md_files)} MD-files and searching for entities in {entities_folder.name}")
            
            all_documents = []
            
            for md_file in tqdm(md_files, desc="Processing MD + Entities"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                article_id = md_file.stem 
                
                entities_list = []
                entity_file = entities_folder / f"{article_id}.json"
                
                if entity_file.exists():
                    try:
                        with open(entity_file, 'r', encoding='utf-8') as f:
                            ent_data = json.load(f)
                            entities_list = ent_data.get("entities", [])
                    except Exception as e:
                        print(f"⚠️ Error reading entities for {article_id}: {e}")
                else:
                    print(f"🔍 Entities for {article_id} not found in {entity_file}")

                doc = Document(
                    page_content=content,
                    metadata={
                        'article_id': article_id,
                        'entities': ', '.join(entities_list), 
                        'entity_count': len(entities_list),
                        'char_count': len(content),
                        'source_type': 'docling_markdown'
                    }
                )
                
                chunks = self.text_splitter.split_documents([doc])
                
                for idx, chunk in enumerate(chunks):
                    chunk.metadata['chunk_index'] = idx
                    chunk.metadata['total_chunks'] = len(chunks)
                    chunk.metadata['preview'] = chunk.page_content[:150] + "..."
                
                all_documents.extend(chunks)
            
            print(f"\n📦 Total chunks: {len(all_documents)}")
            print("🏗️ Indexing in ChromaDB...")
            self.create_vectorstore(all_documents, reset=True)

            print("📝 Updating BM25...")
            self.bm25_docs = [doc.page_content for doc in all_documents]
            self.bm25_metadata = [doc.metadata for doc in all_documents]
            tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized_docs)
            
            bm25_path = self.persist_directory / "bm25_index.pkl"
            with open(bm25_path, 'wb') as f:
                pickle.dump({'bm25': self.bm25, 'docs': self.bm25_docs, 'metadata': self.bm25_metadata}, f)
            print("✅ Done! Database with entities saved")

    def create_vectorstore(self, documents: List[Document], reset: bool = False):
        if reset and self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            print("🗑️  Deleted existing vectorstore")
        
        if documents:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory)
            )
            print(f"✅ Created vectorstore with {len(documents)} chunks")
        else:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            print("✅ Loaded existing vectorstore")
        
        return self.vectorstore
    
    def build_from_merged_data(self, merged_json_path: str):
        merged_path = Path(merged_json_path)
        
        if not merged_path.exists():
            raise FileNotFoundError(f"File not found: {merged_json_path}")
        
        print(f"\n📚 Loading data from: {merged_path.name}")
        
        with open(merged_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"📄 Processing {len(articles)} articles\n")
        
        all_documents = []
        
        for article in tqdm(articles, desc="Processing articles"):
            doc = Document(
                page_content=article['full_text'],
                metadata={
                    'article_id': article['article_id'],
                    'entities': ', '.join(article['entities']),
                    'entity_count': len(article['entities']),
                    'char_count': article['metadata']['char_count'],
                    'llm_cleaned': article['metadata']['llm_cleaned']
                }
            )
            
            chunks = self.text_splitter.split_documents([doc])
            
            for idx, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = idx
                chunk.metadata['total_chunks'] = len(chunks)
                chunk.metadata['preview'] = chunk.page_content[:150] + "..."
            
            all_documents.extend(chunks)
        
        print(f"\n📦 Total chunks created: {len(all_documents)}")

        print("Creating vector store...")
        self.create_vectorstore(all_documents, reset=True)

        print("📝 Building BM25 index...")
        self.bm25_docs = [doc.page_content for doc in all_documents]
        self.bm25_metadata = [doc.metadata for doc in all_documents]
        
        tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        bm25_path = self.persist_directory / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'docs': self.bm25_docs,
                'metadata': self.bm25_metadata
            }, f)
        print("✅ BM25 index saved")
        
        print("="*70)
        print("📊 KNOWLEDGE BASE STATISTICS")
        print("="*70)
        print(f"✅ Articles: {len(articles)}")
        print(f"📦 Chunks: {len(all_documents)}")
        print(f"🔍 BM25 index: {len(self.bm25_docs)} documents")
        print(f"💾 Saved to: {self.persist_directory}")
        print("="*70)
        
        return self.vectorstore
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run build_from_merged_data first.")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def load_bm25(self):
        bm25_path = self.persist_directory / "bm25_index.pkl"
        print(f"🔍 Checking for BM25 index at: {bm25_path.absolute()}")
        
        if bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.bm25_docs = data['docs']
                    self.bm25_metadata = data['metadata']
                print("✅ Loaded BM25 index successfully")
            except Exception as e:
                print(f"❌ Error loading BM25 pickle: {e}")
                self.bm25 = None
        else:
            print("⚠️ BM25 file NOT found. Hybrid search will fail.")
            self.bm25 = None
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7, filter_entities: List[str] = None) -> List[Document]:
        vector_docs = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), 
                           key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        bm25_results = [
            (Document(page_content=self.bm25_docs[i], 
                     metadata=self.bm25_metadata[i]), bm25_scores[i])
            for i in top_indices
        ]
        
        combined = {}
        
        max_v = max(s for _, s in vector_docs) if vector_docs else 1.0
        max_b = max(s for _, s in bm25_results) if bm25_results else 1.0
        
        for doc, score in vector_docs:
            doc_id = f"{doc.metadata['article_id']}_chunk_{doc.metadata['chunk_index']}"
            norm_score = 1 - (score / max_v) 
            combined[doc_id] = {
                'doc': doc, 
                'score': alpha * norm_score
            }

        for doc, score in bm25_results:
            doc_id = f"{doc.metadata['article_id']}_chunk_{doc.metadata['chunk_index']}"
            norm_score = score / max_b
            if doc_id in combined:
                combined[doc_id]['score'] += (1-alpha) * norm_score
            else:
                combined[doc_id] = {
                    'doc': doc, 
                    'score': (1-alpha) * norm_score
                }

        if filter_entities:
            q_entities_clean = [e.lower().strip() for e in filter_entities]
            
            for doc_id, item in combined.items():
                doc_meta_entities = item['doc'].metadata.get('entities', '').lower()
                
                hits = sum(1 for qe in q_entities_clean if qe in doc_meta_entities)
                
                if hits > 0:
                    boost_factor = 1.2 + (0.05 * (hits - 1))
                    item['score'] *= boost_factor
                    print(f"🚀 Boosted {doc_id} by {boost_factor}x (Matched: {hits})")

        ranked = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in ranked[:k]]
    
    def get_hybrid_retriever(self, k: int = 5, alpha: float = 0.7):
        
        kb = self
        
        class HybridRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager=None):
                return kb.hybrid_search(query, k=k, alpha=alpha)
        
        return HybridRetriever()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "data_processing"
    CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db" 
    print("\n" + "="*70)
    print("🔨 BUILDING KB: WITH LLM CLEANING")
    print("="*70)
    
    kb_llm = AlzheimerKnowledgeBase(
        collection_name="alzheimer_papers_llm",
        persist_directory=str(CHROMA_DB_DIR / "with_llm"),
        embedding_model="NeuML/pubmedbert-base-embeddings"
    )
    
    kb_llm.build_from_merged_data(
        merged_json_path=DATA_DIR / "merged" / "with_llm" / "all_merged_llm.json"
    )
    
    print("\n" + "="*70)
    print("🔨 BUILDING KB: WITHOUT LLM CLEANING")
    print("="*70)
    
    kb_basic = AlzheimerKnowledgeBase(
        collection_name="alzheimer_papers_basic",
        persist_directory=str(CHROMA_DB_DIR / "no_llm"),
        embedding_model="NeuML/pubmedbert-base-embeddings"
    )
    
    kb_basic.build_from_merged_data(
        merged_json_path=DATA_DIR / "merged" / "no_llm" / "all_merged_basic.json"
    )
    
    print("\n" + "="*70)
    print("🔍 TEST RETRIEVAL")
    print("="*70)
    
    retriever = kb_llm.get_retriever(search_type="similarity", k=3)
    
    test_query = "What are the main protein targets for Alzheimer's disease treatment?"
    docs = retriever.invoke(test_query)
    
    print(f"\nQuery: {test_query}\n")
    
    for idx, doc in enumerate(docs, 1):
        print(f"Result {idx}:")
        print(f"  Article: {doc.metadata['article_id']}")
        print(f"  Chunk: {doc.metadata['chunk_index']}/{doc.metadata['total_chunks']}")
        print(f"  Entities: {doc.metadata['entities'][:80]}...")
        print(f"  Text: {doc.page_content[:200]}...")
        print()