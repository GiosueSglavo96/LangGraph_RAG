from src.multi_agent_system.nodes.agents.ethics_agent import check_query_ethics
from src.multi_agent_system.nodes.functions.not_etchis_handler import not_ethics_handler
from src.multi_agent_system.nodes.agents.web_search_agent import web_search_agent
from src.multi_agent_system.nodes.functions.ethics_router import ethics_router

from src.multi_agent_system.state.workflow_state import AppState

from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(AppState)

graph_builder.add_node("CheckEthics", check_query_ethics)
graph_builder.add_node("NotEthics", not_ethics_handler)
graph_builder.add_node("WebSearch", web_search_agent)

graph_builder.add_edge(START, "CheckEthics")
graph_builder.add_conditional_edges("CheckEthics", 
                                   ethics_router, 
                                   {
                                       "Unethical": "NotEthics",
                                       "Ethical": "WebSearch"
                                   })

graph_builder.add_edge("NotEthics", END)
graph_builder.add_edge("WebSearch", END)