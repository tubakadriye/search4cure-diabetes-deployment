from langchain_google_genai import ChatGoogleGenerativeAI

def get_gemini_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )
