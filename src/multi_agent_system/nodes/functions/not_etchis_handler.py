from src.multi_agent_system.state.workflow_state import AppState

def not_ethics_handler(state: AppState) -> AppState:
    not_ethics_reason = state.ethics.reason
    state.response = not_ethics_reason
    return state
