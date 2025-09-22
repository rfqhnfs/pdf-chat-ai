import tiktoken

class TokenCount:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback encoding if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def num_tokens_from_string(self, text):
        """Count tokens in a string or convert dict/object to string first"""
        if isinstance(text, dict):
            text = str(text)
        elif not isinstance(text, str):
            text = str(text)
        
        try:
            return len(self.encoding.encode(text))
        except:
            # Fallback token estimation
            return len(str(text).split()) * 1.3  # Rough estimate
