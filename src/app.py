import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from rag_agent import AlzheimerRAGAgent

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "chromadb" / "docling_with_entities"

st.set_page_config(
    page_title="Alzheimer's Target Discovery (Docling Edition)",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    .stChatFloatingInputContainer {bottom: 20px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Search Settings")
    
    use_hybrid = st.toggle("Hybrid Search (Vector + BM25)", value=True)
    k_results = st.number_input("Number of chunks (K)", min_value=1, max_value=20, value=5)
    alpha = st.slider("Balance (Vector <-> BM25)", 0.0, 1.0, 0.7, help="1.0 = vector only, 0.0 = keywords only")
    
    st.divider()
    model_name = st.text_input("МLLM", value="google/gemma-3-27b-it:free")
    use_hyde = st.toggle("Use HyDE", value=True)
    
    st.info(f"📂 Database: `{DB_PATH.name}`")

@st.cache_resource(show_spinner=False)
def get_agent(_path_str, model, hybrid, k, a, hyde):
    return AlzheimerRAGAgent(
        kb_path=_path_str,
        model_name=model,
        use_hybrid_search=hybrid,
        k_results=k,
        alpha=a,
        use_hyde=hyde
    )

st.title("🧬 Alzheimer's Disease Discovery")
st.caption("RAG system based on scientific articles")

refusal_phrase = "I am a specialized assistant for Alzheimer's research"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"] and refusal_phrase not in message["content"]:
            with st.expander("📚 References"):
                for s in message["sources"]:
                    st.write(f"**{s['article_id']}** (Chunk {s['chunk_index']})")
                    st.caption(s['preview'])

if prompt := st.chat_input("What is APOE3?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer & Analyzing literature"):
            try:
                agent = get_agent(str(DB_PATH), model_name, use_hybrid, k_results, alpha, use_hyde)
                response = agent.query(prompt, return_sources=True)
                
                answer = response["answer"]
                sources = response["sources"]
                
                st.markdown(answer)
                
                if sources and refusal_phrase not in answer:
                    st.markdown("### References")
                    for idx, src in enumerate(sources, 1):
                        with st.expander(f"[{idx}] {src['article_id']} — details"):
                            st.markdown(f"**Found entities:** `{src['entities']}`")
                            st.info(src['full_text'])
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                
            except Exception as e:
                st.error(f"Ошибка: {e}")
