from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import gc

def process_pdf_from_file(file_path, api_key):
    """
    Direct RAG without vector database - 100% reliable for Streamlit Cloud
    """
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Clean text
        cleaned_text = ""
        for doc in documents:
            clean_content = re.sub(r'[\n\xa0\s]+', ' ', doc.page_content).strip()
            clean_content = re.sub(r'\u200b', '', clean_content).strip()
            cleaned_text += clean_content + " "
        
        # Limit text length for memory efficiency
        if len(cleaned_text) > 15000:
            cleaned_text = cleaned_text[:15000]
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0,
            max_tokens=300,
            google_api_key=api_key
        )
        
        # Create simple RAG function
        def simple_rag(question):
            prompt = f"""
            Based on the following document content, answer the question accurately and concisely.
            
            Document Content:
            {cleaned_text}
            
            Question: {question}
            
            Answer based only on the document content above. If the information is not in the document, say "Information not found in the document."
            """
            
            try:
                response = llm.invoke(prompt)
                return {"answer": response.content}
            except Exception as e:
                return {"answer": f"Error processing question: {str(e)}"}
        
        # Clean up memory
        del documents
        gc.collect()
        
        # Return function that mimics langchain invoke
        class SimpleRAGChain:
            def invoke(self, input_dict):
                return simple_rag(input_dict["input"])
        
        return SimpleRAGChain()
        
    except Exception as e:
        gc.collect()
        raise e
