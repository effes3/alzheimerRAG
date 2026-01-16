import streamlit as st
from pathlib import Path
import sys
from rag_agent import AlzheimerRAGAgent
import os
from dotenv import load_dotenv

load_dotenv()

current_file = Path(__file__).resolve()
current_dir = current_file.parent     
project_root = current_dir.parent 
sys.path.append(str(current_dir))

st.set_page_config(
    page_title="Alzheimer's Target Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .element-container img {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    kb_version = st.radio(
        "Knowledge Base Version:",
        ["with_llm (Cleaned)", "no_llm (Raw)"],
        index=0
    )
    
    kb_folder = "with_llm" if "with_llm" in kb_version else "no_llm"
    kb_path = project_root / "data" / "chromadb" / kb_folder
    
    st.divider()
    
    use_hybrid = st.toggle("Use Hybrid Search (Vector + BM25)", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        k_results = st.number_input("Top K", min_value=1, max_value=20, value=5)
    with col2:
        alpha = st.slider("Alpha (Vec/BM25)", 0.0, 1.0, 0.7, 0.1, help="1.0 = Vector only, 0.0 = Keyword only")
        
    default_model = os.getenv("RAG_MODEL_NAME", "google/gemma-3-27b-it:free")
    model_name = st.text_input("LLM Model", value=default_model)
    
    st.info(f"📂 Connected to: `chromadb/{kb_folder}`")

    use_hyde = st.toggle("Use HyDE (Hypothetical Document)", value=True)

@st.cache_resource(show_spinner=False)
def get_agent(path_str, model, hybrid, k, a, hyde):
    return AlzheimerRAGAgent(
        kb_path=path_str,
        model_name=model,
        use_hybrid_search=hybrid,
        k_results=k,
        alpha=a,
        use_hyde=hyde
    )

st.title("🧬 Alzheimer's Disease Drug Target Discovery")
st.markdown("Identifies potential therapeutic targets from scientific literature using RAG")
refusal_phrase = "I am a specialized assistant for Alzheimer's research and can only answer questions related to this domain."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message['sources']:
            if refusal_phrase not in message['content']:
                with st.expander("📚 View Sources used for this answer"):
                    for s in message["sources"]:
                        st.markdown(f"**{s['article_id']}** (Chunk {s.get('chunk_index', '?')})")
                        st.caption(s.get('preview', 'No preview'))

if prompt := st.chat_input("Ask about drug targets (e.g., 'Alzheimer drug targets')"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Searching literature & generating answer..."):
                agent = get_agent(str(kb_path), model_name, use_hybrid, k_results, alpha, use_hyde)
                
                response = agent.query(prompt, return_sources=True)
                answer_text = response["answer"]
                sources = response["sources"]

            message_placeholder.markdown(answer_text)

            if sources and (refusal_phrase not in answer_text):
                st.markdown("### References")
                for idx, src in enumerate(sources, 1):
                    chunk_info = f"Chunk {src.get('chunk_index', '?')}"
                    title = f"**{idx}. {src.get('article_id', 'Unknown Article')}** — *{chunk_info}*"
                    
                    with st.expander(f"Reference {idx}: {src.get('article_id', 'Article')}"):
                        st.markdown(f"**Entities found:** `{src.get('entities', 'None')}`")
                        st.markdown("**Context snippet:**")
                        st.info(src.get('preview', 'No text available')[:400] + "...")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer_text,
                "sources": sources 
            })
            
        except Exception as e:
            st.error(f"**Error:** {str(e)}", icon="🚨")
            print(f"Error details: {e}")