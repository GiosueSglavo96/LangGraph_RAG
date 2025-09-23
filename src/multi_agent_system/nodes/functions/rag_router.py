from src.multi_agent_system.state.workflow_state import AppState

def rag_router(state: AppState) -> AppState:
    print("RAG routing...")
    if state.rag.found_info:
        return "END"
    else:
        state.response.append("I'm sorry, but I don't have the information you're looking for in the loaded documents.\n I try to search on the web for you...\n")
        return "Web Search"