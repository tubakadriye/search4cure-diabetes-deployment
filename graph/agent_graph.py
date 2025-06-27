from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from db.checkpoint.mongo_checkpoint_saver import MongoDBSaver
from db.mongodb_client import MONGODB_URI, MONGO_DB
from motor.motor_asyncio import AsyncIOMotorClient
from agent.state import AgentState

    
def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        return {"messages": [result], "sender": name}
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {"messages": [result], "sender": name}

def build_agent_graph(chatbot_node, tool_node):
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "chatbot")

    mongo_client = AsyncIOMotorClient(MONGODB_URI)
    checkpointer = MongoDBSaver(mongo_client, MONGO_DB, "state_store")

    return workflow.compile(checkpointer=checkpointer)
