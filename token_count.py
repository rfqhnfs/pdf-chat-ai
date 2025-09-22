class TokenCount:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
    
    def num_tokens_from_string(self, text):
        """Simple token estimation without tiktoken dependency"""
        if isinstance(text, dict):
            text = str(text)
        elif not isinstance(text, str):
            text = str(text)
        
        # Simple estimation: ~4 characters per token
        return len(str(text)) // 4
