"""LangChain RunnableWithMessageHistory bridge for in-process session memory."""
from __future__ import annotations

def build_langchain_memory_chain():
    """Return a RunnableWithMessageHistory chain plus its session history store."""
    try:
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
    except ImportError as e:
        raise RuntimeError("Install langchain-core from requirements.txt") from e
    store={}
    def get_history(session_id: str):
        if session_id not in store: store[session_id]=InMemoryChatMessageHistory()
        return store[session_id]
    chain=RunnableLambda(lambda values: values["input"])
    wrapped=RunnableWithMessageHistory(chain,get_history,input_messages_key="input",history_messages_key="history")
    return wrapped,store
