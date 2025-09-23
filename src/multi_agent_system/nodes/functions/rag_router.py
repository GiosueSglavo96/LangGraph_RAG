from src.multi_agent_system.state.workflow_state import AppState

def rag_router(state: AppState) -> AppState:
    print("RAG routing...")
    if state.rag.found_info:
        return "END"
    else:
        return "Web Search"