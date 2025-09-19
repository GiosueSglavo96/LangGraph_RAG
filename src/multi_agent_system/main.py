from src.multi_agent_system.graph.graph import graph_builder

rag_config_dict = {
    "chunk_size": None,
    "chunk_overlap": None,
    "top_n_semantic": None,
    "top_n_text": None,
    "alpha": None,
    "text_boost": None,
    "final_top_k": None,
    "mmr": None,
    "mmr_lambda": None,
}

def run_workflow(query: str, rag_config: dict):
    
    for key in rag_config:
        if key in rag_config:
            rag_config_dict[key] = rag_config[key]
    
    
    #if any(value is None for value in rag_config_dict.values()):
    #   raise ValueError("Error: All RAG configuration parameters must be set.")
    
    print("RAG CONFIG IN RUN WORKFLOW:", rag_config_dict)

    graph = graph_builder.compile()

    graph_response = graph.invoke({"query": query})
    response = graph_response["response"]

    return response