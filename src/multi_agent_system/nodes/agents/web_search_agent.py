from src.multi_agent_system.models.llm_model import get_llm
from src.multi_agent_system.state.workflow_state import AppState, QueryEthics

def web_search_agent(state: AppState) -> AppState:
    print("Web Search Agent Invoked")
    return state