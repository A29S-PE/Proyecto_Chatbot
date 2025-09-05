from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

class MemoryManager:
    def __init__(self):
        self.user_memories = {}

    def get_memory(self, user_id: str):
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferWindowMemory(
                memory_key="history", 
                return_messages=True, 
                k=5
            )
        return self.user_memories[user_id]


def format_history(history_msgs):
    if len(history_msgs) != 0:
        formatted = "Estos son los últimos mensajes entre el usuario y tú:\n"
        for msg in history_msgs:
            if isinstance(msg, HumanMessage):
                formatted += f"user: '{msg.content}'\n"
            elif isinstance(msg, AIMessage):
                formatted += f"assistant: '{msg.content}'\n"
            else:
                formatted += f"{msg.type}: '{msg.content}'\n"
    else:
        formatted = ""
    return formatted

