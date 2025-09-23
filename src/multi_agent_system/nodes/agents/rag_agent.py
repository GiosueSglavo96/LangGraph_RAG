from src.multi_agent_system.models.llm_model import get_llm
from src.multi_agent_system.state.workflow_state import AppState, RagState
from src.multi_agent_system.tools.rag_tool import execute_rag
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful medical assistant.
        You must answer the user’s questions **only** using the information provided in the context below.
        Always respond in structured JSON format with your assessment.
        - If the context contains relevant information, set:
            - The 'found_info' variable to True.
            - The 'context' variable to the context passed in the prompt.
            - The 'response' variable to a concise and accurate answer to the query based solely on the context. 
            Always include in the response, the sources of the information as they appear in the context, in the format: [source: <file_name>] where <file_name> is the name of the file containing the information without the complete path.
        - If the context does NOT contain any relevant information, set:
            - The 'found_info' variable to False.
            - The 'context' to a empty string.
            - The 'response' variable to "I'm sorry, but I don't have the information you're looking for."""),
        ("human","""
         Query: \n{query}\n\n
         Context: \n{context}\n\n
         Instructions: \n
         - Do NOT use any knowledge outside the given context.
         - Do NOT guess or fabricate information.
         """)
    ]
)

def medical_rag(state: AppState) -> AppState:
    print("Executing medical_rag agent...\n")
    query = state.query
    try:
        context = execute_rag(query)

        llm = get_llm()

        structured_llm = llm.with_structured_output(RagState)
        rag_chain = prompt_template | structured_llm
        rag_response = rag_chain.invoke({"query": query, "context": context})
        state.rag = rag_response
        state.response.append(state.rag.response)
        return state
    except Exception as e:
        print(f"Error in RAG execution: {e}")
        # Fallback: empty RAG state if there's an error
        state.rag = RagState(
            found_info=False,
            context="",
            response="Error during RAG execution: " + str(e)
        )
        return state