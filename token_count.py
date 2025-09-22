class TokenCount:
    def __init__(self, model_name="gpt-3.5-turbo"):
        pass
    
    def num_tokens_from_string(self, text):
        """Simple token estimation"""
        return len(str(text)) // 4
