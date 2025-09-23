from src.multi_agent_system.state.workflow_state import AppState

def medicine_router(state: AppState) -> AppState:
    print("Medicine routing...")
    if state.medicine.is_medicine_related:
        return "Executing RAG" 
    else:
        return "Web Search"