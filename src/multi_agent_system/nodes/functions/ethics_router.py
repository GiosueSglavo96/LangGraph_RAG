from src.multi_agent_system.state.workflow_state import AppState

def ethics_router(state: AppState) -> AppState:
    print("Ethical routing...")
    if state.ethics.is_ethical:
        return "Ethical" 
    else:
        return "Unethical"