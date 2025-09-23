from src.multi_agent_system.state.workflow_state import AppState

def medicine_router(state: AppState) -> AppState:
    print("Medicine routing...")
    if state.medicine.is_medicine_related:
        state.response.append("The query seems to be related to medicine.\n I try to answer using RAG on the loaded medical documents\n")
        return "Executing RAG" 
    else:
        state.response.append("The query seems to be unrelated to medicine.\n I try to answer using a web search.\n")
        return "Web Search"