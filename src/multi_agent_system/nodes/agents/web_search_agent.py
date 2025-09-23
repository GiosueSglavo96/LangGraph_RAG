from src.multi_agent_system.models.llm_model import get_llm
from src.multi_agent_system.state.workflow_state import AppState, QueryEthics
from src.multi_agent_system.tools.weather_tool import get_weather

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_tavily import TavilySearch
from datetime import datetime, timedelta

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
        You are an assistant that processes user queries by deciding which tool to invoke based on the query content.
        
        IMPORTANT: Today's date is {datetime.now().strftime('%Y-%m-%d')}. Always use current dates!
        
        If the user's query is about weather forecasts, you must call the custom tool get_weather.
        The get_weather tool takes three parameters: location, start_date, end_date.
        
        DATE HANDLING RULES:
        - If user says "today": use {datetime.now().strftime('%Y-%m-%d')} for both start_date and end_date
        - If user says "tomorrow": use {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} for both start_date and end_date  
        - If user says "this week": use {datetime.now().strftime('%Y-%m-%d')} as start_date and {(datetime.now() + timedelta(days=6)).strftime('%Y-%m-%d')} as end_date
        - If user doesn't specify a period: default to today ({datetime.now().strftime('%Y-%m-%d')}) for both start_date and end_date
        - Always use YYYY-MM-DD format for dates
        - Never use dates from previous years!
        
        The user must give you a location (city name) and optionally a period to get the weather forecast.
        If the weather tool is used, use the returned data to compose a clear weather report for the user.
        
        For all other general queries that are not related to weather forecasts, you must call the web search tool called tavily_search.
        Make your decision based only on the content of the query, and always pick exactly one tool per query.
        Do not provide any answer yourself; instead, return only the invocation of the chosen tool."""),
        ("human", "{query}")
    ]
)

def web_search_agent(state: AppState) -> AppState:
    print("Web Search Agent Invoked")
    query = state.query
    try:
        llm = get_llm()

        tavily_search_tool = TavilySearch(
            max_results=2,
            topic="general",
            include_raw_content=True)

        tools = [get_weather, tavily_search_tool]

        llm_with_tools = llm.bind_tools(tools)
        web_search_chain = prompt_template | llm_with_tools
        ai_message = web_search_chain.invoke({"query": query})

        # Se non ci sono tool_calls, è un errore (non dovrebbe accadere col tuo prompt)
        if not ai_message.tool_calls:
            state.response.append("No tool was invoked.")
            return state

        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "tavily_search":
            state.response.append("Invoking Tavily web search tool...\n")
            tavily_response = tavily_search_tool.invoke(tool_args)
            # Crea una sequenza di messaggi per il secondo invoke
            messages = [
                HumanMessage(content=query),
                ai_message,  # Il messaggio AI con la tool call
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(tavily_response)
                )
            ]
            final_response = llm_with_tools.invoke(messages)
            
        elif tool_name == "get_weather":
            state.response.append("Invoking the weather tool...\n")
            weather_response = get_weather.invoke(tool_args)
            
            # Crea una sequenza di messaggi per il secondo invoke
            messages = [
                HumanMessage(content=query),
                ai_message,  # Il messaggio AI con la tool call
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    content=str(weather_response)
                )
            ]
            final_response = llm_with_tools.invoke(messages)

        state.response.append(final_response.content)
    except Exception as e:
        print(f"Error in web search agent: {e}")
        state.response.append(f"Error during web search: {str(e)}")
    return state