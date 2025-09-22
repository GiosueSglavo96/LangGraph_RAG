import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from time import time
import streamlit as st

from src.multi_agent_system.main import run_workflow

UPLOADED_FILES_DIRECTORY = "../data/input_files"
os.makedirs(UPLOADED_FILES_DIRECTORY, exist_ok=True)

alpha = 0.75
text_boost = 0.2

if "messages" not in st.session_state:
    st.session_state.messages = []

if "configured_rag" not in st.session_state:
    st.session_state.configured_rag = True

if "rag_config" not in st.session_state:
    st.session_state.rag_config = {}

st.title("RAG & LangGraph Application")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(response):
    for word in response.split():
        yield word + " "

with st.sidebar:
    st.header("RAG Settings")

    with st.container(border=True):
        st.write("Do you want to enable MMR?")
        mmr_on = st.segmented_control("MMR", 
                                    options=["Enable MMR", "Disable MMR"], 
                                    default="Disable MMR", 
                                    key="mmr_control",
                                    selection_mode="single", 
                                    width="stretch",
                                    label_visibility="collapsed")

    with st.container(border=True):
        st.write("What kind of search do you want to perform?")
        search_type = st.segmented_control("Search Type", 
                                    options=["Semantic Search", "Hybrid Search", "Text Search"], 
                                    default="Hybrid Search", 
                                    key="search_type",
                                    selection_mode="single", 
                                    width="stretch",
                                    label_visibility="collapsed")
        
        if search_type == "Semantic Search":
            alpha = 1.0
            text_boost = 0.0
        elif search_type == "Text Search":
            alpha = 0.0
            text_boost = 1.0

    with st.form("rag_form"):
        st.subheader("Files Upload")
        uploaded_files = st.file_uploader("Upload the context files", 
                                          accept_multiple_files=True, 
                                          type=["pdf", "csv", "txt"])
        
        st.subheader("Splitting settings")
        chunk_size = st.slider("Chunk Size", 
                               min_value=50, 
                               max_value=1500, 
                               value=700, 
                               step=50, 
                               key="chunk_size",
                               help="Size of each text chunk in characters.")
        chunk_overlap = st.slider("Chunk Overlap", 
                                  min_value=0, 
                                  max_value=500, 
                                  value=50, 
                                  step=10, 
                                  key="chunk_overlap",
                                  help="Number of overlapping characters between consecutive chunks.")
        st.info("Ensure that chunk overlap is less than chunk size to avoid errors.")


        st.subheader("Search settings")
        
        options = ["Full Semantic Search", "Hybrid Search", "Full Text Search"]
       
        top_N_semantic = st.slider("Top N Semantic", 
                                   min_value=10, 
                                   max_value=300, 
                                   value=30, 
                                   step=5, 
                                   key="top_n_semantic",
                                   help="Number of top similar chunks to retrieve based on semantic similarity.")
        
        top_N_text = st.slider("Top N Text", 
                              min_value=10, 
                              max_value=300, 
                              value=30, 
                              step=5, 
                              key="top_n_text",
                              help="Number of top similar chunks to retrive based on text matching.")
        
        alpha = st.slider("Alpha", 
                         min_value=0.0, 
                         max_value=1.0, 
                         value=alpha, 
                         step=0.1, 
                         key="alpha",
                         help="""Weighting factor for semantic similarity:
                              Alpha Parameter Behavior:
                              - alpha = 0.0: Pure text-based ranking (BM25, keyword matching)
                              - alpha = 0.5: Equal weight for semantic and text relevance
                              - alpha = 0.75: Semantic similarity prioritized (current setting)
                              - alpha = 1.0: Pure semantic ranking (cosine similarity only)
                                
                              Use Case Recommendations:
                              - Technical queries: 0.7-0.9 (semantic understanding important)
                              - Factual queries: 0.5-0.7 (balanced approach)
                              - Keyword searches: 0.3-0.5 (text matching more important)
                              - Conversational queries: 0.6-0.8 (semantic context matters)""")
        
        text_boost = st.slider("Text Boost", 
                               min_value=0.0, 
                               max_value=1.0, 
                               value=text_boost, 
                               step=0.1, 
                               key="text_boost",
                               help="""Boost factor for text matching score to increase its influence\n.
                                    Boost Value Guidelines:\n
                                    - Low boost (0.1-0.2): Subtle preference for hybrid matches\n
                                    - Medium boost (0.2-0.4): Strong preference for hybrid matches\n
                                    - High boost (0.5+): Heavy preference, may dominate ranking\n
                                    Optimal Settings:\n
                                    - General use: 0.15-0.25\n
                                    - Technical content: 0.20-0.30\n
                                    - Factual queries: 0.10-0.20""")
        
        final_top_k = st.slider("Final Top K", 
                                min_value=1, 
                                max_value=30, 
                                value=5, 
                                step=1, 
                                key="final_top_k",
                                help="Number of final top chunks to return after combining semantic and text search (Hybrid Search).")
        
        
        if mmr_on == "Enable MMR":
            mmr = True
            st.subheader("MMR Settings")
            mmr_lambda = st.slider("MMR Lambda", 
                                   min_value=0.0, 
                                   max_value=1.0, 
                                   value=0.5, 
                                   step=0.1, 
                                   key="mmr_lambda",
                                   help="""Lambda parameter for MMR (Maximal Marginal Relevance):
                                         0.0 = only relevance,
                                         0.5 = balanced between relevance and diversity,
                                         1.0 = only diversity.""")
    
        else:
            mmr = False
            mmr_lambda = None
        
        submitted = st.form_submit_button("Submit", width="stretch")
        if submitted:

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    saving_file_path = os.path.join(UPLOADED_FILES_DIRECTORY, uploaded_file.name)
                    
                    if os.path.exists(saving_file_path):
                        st.warning(f"File {uploaded_file.name} already exists and will be skipped.")
                        continue

                    with open(saving_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"File {uploaded_file.name} saved successfully.")
                st.session_state.configured_rag = True

                rag_config = {
                    "chunk_size": st.session_state.chunk_size,
                    "chunk_overlap": st.session_state.chunk_overlap,
                    "top_n_semantic": st.session_state.top_n_semantic,
                    "top_n_text": st.session_state.top_n_text,
                    "alpha": st.session_state.alpha,
                    "text_boost": st.session_state.text_boost,
                    "final_top_k": st.session_state.final_top_k,
                    "mmr": mmr,
                    "mmr_lambda": mmr_lambda,
                }
                st.session_state.rag_config = rag_config

                st.success("RAG configuration submitted!")
            else:
                st.error("Please upload one or more files to proceed with RAG.")
                #st.session_state.configured_rag = False

print(st.session_state.configured_rag)
prompt = st.chat_input("Ask something...", disabled=not st.session_state.configured_rag)
if prompt:
    with st.chat_message("user", width="stretch", avatar="user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", width="stretch", avatar="assistant"):
        response = run_workflow(prompt, st.session_state.rag_config)
        print("STREAMLIT RESPONSE:", response)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.info("Please configure RAG settings in the sidebar to enable chat input.")