from langchain.agents import AgentExecutor, create_tool_calling_agent
from llm.gemini_llm import get_gemini_llm
from prompts.agent_prompt import get_agent_prompt
from db.memory import get_conversation_memory
from agent_tools.article_tools import article_page_vector_search_tool, vector_search_image_tool
from agent_tools.csv_tools import create_new_record , csv_files_vector_search_tool

def get_toolbox():
    toolbox = []
    records_embeddings_collection_tools = [
    csv_files_vector_search_tool,
    #hybrid_search_tool,
    create_new_record,
    ]
    docs_multimodal_collection_tools = [article_page_vector_search_tool, vector_search_image_tool]

    toolbox.extend(records_embeddings_collection_tools)
    toolbox.extend(docs_multimodal_collection_tools)
    # Add other tools as needed
    return toolbox

def build_agent_executor():
    llm = get_gemini_llm()
    prompt = get_agent_prompt()
    tools = get_toolbox()
    memory = get_conversation_memory()
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
    )
