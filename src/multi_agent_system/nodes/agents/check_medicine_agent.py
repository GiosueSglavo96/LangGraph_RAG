from src.multi_agent_system.models.llm_model import get_llm
from src.multi_agent_system.state.workflow_state import AppState, QueryMedicine

from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
         Your task is to evaluate user input queries and determine whether they are related to medicine.
         Always respond in structured JSON format with your assessment. 
         If the query is related to medicine - including but not limited to topics such as 
         diseases, treatments, drugs, medicines, symptoms, medical procedures, healthcare professionals, medications, or medical research - 
         then set the "is_medicine_related" field in the output JSON to True. 
         Otherwise, set it to False."""),
         ("human", "{query}")
    ]
)

def check_query_medicine(state: AppState) -> AppState:
    query = state.query
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(QueryMedicine)
        check_medicine_chain = prompt_template | structured_llm
        medicine_response = check_medicine_chain.invoke({"query": query})
        state.medicine = medicine_response
        return state
    except Exception as e:
        print(f"Error in medicine check: {e}")
        # Fallback: mark as non-medicine-related if there's an error
        state.medicine_check = QueryMedicine(
            is_medicine_related=False,
            reason=f"Error during medicine evaluation: {str(e)}"
        )
        return state