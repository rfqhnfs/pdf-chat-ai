from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import re
import gc

def process_pdf_from_file(file_path, api_key):
    """
    Simple text-based RAG without any vector database
    100% compatible with all Streamlit Cloud environments
    """
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Extract and clean all text
        full_text = ""
        for doc in documents:
            clean_content = re.sub(r'[\n\xa0\s]+', ' ', doc.page_content).strip()
            clean_content = re.sub(r'\u200b', '', clean_content).strip()
            full_text += clean_content + " "
        
        # Limit text for memory efficiency (Streamlit Cloud limit)
        if len(full_text) > 12000:
            full_text = full_text[:12000]
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0,
            max_tokens=300,
            google_api_key=api_key
        )
        
        # Simple RAG function
        def answer_question(question):
            prompt = f"""
You are an AI assistant that answers questions based on document content.

Document Content:
{full_text}

Question: {question}

Instructions:
- Answer based ONLY on the document content above
- Be accurate and concise
- If information is not in the document, say "Information not found in the document"
- For insurance documents, look for RCV, ACV, depreciation amounts, claim details, etc.

Answer:"""
            
            try:
                response = llm.invoke(prompt)
                return {"answer": response.content}
            except Exception as e:
                return {"answer": f"Error: {str(e)}"}
        
        # Clean up memory
        del documents
        gc.collect()
        
        # Return chain-like object
        class TextRAGChain:
            def invoke(self, input_dict):
                return answer_question(input_dict["input"])
        
        return TextRAGChain()
        
    except Exception as e:
        gc.collect()
        raise Exception(f"PDF processing failed: {str(e)}")
