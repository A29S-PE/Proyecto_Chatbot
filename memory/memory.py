from langchain.memory import ConversationBufferMemory

class MemoryManager:
    def __init__(self):
        self.user_memories = {}

    def get_memory(self, user_id: str):
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
        return self.user_memories[user_id]
