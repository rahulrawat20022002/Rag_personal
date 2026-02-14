from langchain_community.chat_message_histories import ChatMessageHistory

class MemoryAgent:
    def __init__(self):
        # Initialize an empty history in RAM
        self.history = ChatMessageHistory()

    def add_user_message(self, message: str):
        """Saves the user's input."""
        self.history.add_user_message(message)

    def add_ai_message(self, message: str):
        """Saves the AI's response."""
        self.history.add_ai_message(message)

    def get_history_string(self) -> str:
        """
        Converts the list of messages into a readable string for the LLM.
        Format:
        Human: ...
        AI: ...
        """
        messages = self.history.messages
        if not messages:
            return "No previous conversation."
            
        formatted_history = []
        for msg in messages:
            role = "Human" if msg.type == "human" else "AI"
            formatted_history.append(f"{role}: {msg.content}")
            
        return "\n".join(formatted_history)

    def clear(self):
        """Wipes the memory (Optional)."""
        self.history.clear()