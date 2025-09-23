from src.multi_agent_system.nodes.agents.check_ethics_agent import check_query_ethics
from src.multi_agent_system.nodes.functions.ethics_router import ethics_router
from src.multi_agent_system.nodes.functions.not_etchis_handler import not_ethics_handler
from src.multi_agent_system.nodes.agents.check_medicine_agent import check_query_medicine
from src.multi_agent_system.nodes.functions.medicine_router import medicine_router
from src.multi_agent_system.nodes.agents.rag_agent import medical_rag
from src.multi_agent_system.nodes.functions.rag_router import rag_router
from src.multi_agent_system.nodes.agents.web_search_agent import web_search_agent

from src.multi_agent_system.state.workflow_state import AppState

from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(AppState)

graph_builder.add_node("CheckEthics", check_query_ethics)
graph_builder.add_node("NotEthics", not_ethics_handler)
graph_builder.add_node("CheckMedicine", check_query_medicine)
graph_builder.add_node("Rag", medical_rag)
graph_builder.add_node("WebSearch", web_search_agent)

graph_builder.add_edge(START, "CheckEthics")

graph_builder.add_conditional_edges("CheckEthics", 
                                   ethics_router, 
                                   {
                                       "Unethical": "NotEthics",
                                       "Ethical": "CheckMedicine"
                                   })

graph_builder.add_conditional_edges("CheckMedicine",
                                      medicine_router,
                                      {
                                        "Web Search": "WebSearch",
                                        "Executing RAG": "Rag"
                                      })

graph_builder.add_conditional_edges("Rag",
                                   rag_router,
                                   {
                                       "END": END,
                                       "Web Search": "WebSearch"
                                   })

graph_builder.add_edge("WebSearch", END)
graph_builder.add_edge("NotEthics", END)