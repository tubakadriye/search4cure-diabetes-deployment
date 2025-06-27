from langchain.memory import ConversationBufferMemory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from db.mongodb_client import MONGODB_URI, MONGO_DB  # Or however you handle config


def get_session_history(session_id: str):
    return MongoDBChatMessageHistory(
        MONGODB_URI, session_id, database_name=MONGO_DB, collection_name="history"
    )

def get_conversation_memory(session_id: str = "latest_agent_session"):
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=get_session_history(session_id)
    )
