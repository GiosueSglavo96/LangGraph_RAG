from agents.ethics_agent import check_query_ethics
from state.workflow_state import AppState

from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(AppState)

graph_builder.add_node("CheckEthics", check_query_ethics)

graph_builder.add_edge(START, "CheckEthics")
graph_builder.add_edge("CheckEthics", END)