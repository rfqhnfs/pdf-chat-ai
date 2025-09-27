from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
import re
import gc
import os

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

def extract_glossary_text_pypdf2(glossary_path):
    """Extract text from insurance glossary PDF using PyPDF2 (no LangChain)"""
    try:
        print(f"[DEBUG] Extracting glossary text using PyPDF2: {glossary_path}")
        
        # Check if file exists first
        if not os.path.exists(glossary_path):
            print(f"[ERROR] File does not exist: {glossary_path}")
            return None
        
        # Extract text from PDF using PyPDF2
        glossary_text = ""
        with open(glossary_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"[DEBUG] PDF loaded: {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    glossary_text += page_text + "\n\n"
                    print(f"[DEBUG] Page {page_num + 1}: {len(page_text)} characters extracted")
        
        if not glossary_text.strip():
            print("[ERROR] No text extracted from PDF")
            return None
        
        # Clean text
        glossary_text = re.sub(r'[\n\xa0\s]+', ' ', glossary_text).strip()
        glossary_text = re.sub(r'\u200b', '', glossary_text).strip()
        
        print(f"[SUCCESS] Total glossary text extracted: {len(glossary_text)} characters")
        return glossary_text
        
    except Exception as e:
        print(f"[ERROR] Exception in extract_glossary_text_pypdf2: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return None

def create_smart_claim_system_langchain(user_pdf_path, glossary_text, api_key):
    """Create smart system using LangChain (your original working approach)"""
    try:
        print(f"[DEBUG] Creating smart claim system with LangChain")
        
        # Load user PDF with LangChain (your original approach)
        loader = PyPDFLoader(user_pdf_path)
        documents = loader.load()

        # Extract and clean user text
        user_text = ""
        for doc in documents:
            clean_content = re.sub(r'[\n\xa0\s]+', ' ', doc.page_content).strip()
            clean_content = re.sub(r'\u200b', '', clean_content).strip()
            user_text += clean_content + " "

        # Limit text for memory efficiency
        if len(user_text) > 10000:
            user_text = user_text[:10000]
        
        # Limit glossary text
        if len(glossary_text) > 5000:
            glossary_text = glossary_text[:5000]

        print(f"[DEBUG] Content lengths - User claim: {len(user_text)} chars, Glossary: {len(glossary_text)} chars")
        
        # Create LLM (your original working setup)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.2,
            max_tokens=400,
            google_api_key=api_key
        )

        def smart_answer(question):
            """Use both user document and glossary to provide smart answers"""
            
            # Create enhanced prompt that uses both documents
            prompt = f"""You are an expert insurance advisor helping customers understand their insurance claims.

USER'S INSURANCE CLAIM DOCUMENT:
{user_text}

INSURANCE TERMS GLOSSARY:
{glossary_text}

Question: {question}

Instructions:
1. **Find specific information** from the USER'S CLAIM DOCUMENT (amounts, dates, policy details, etc.)
2. **Use the GLOSSARY** to explain any insurance terminology in simple, clear language
3. **Combine both** to give a complete, educational answer

Your answer should:
- Start with specific information from the claim document (if available)
- Explain insurance terms using the glossary definitions
- Be educational and easy to understand
- If information isn't available, clearly state that

Example format for "What is my ACV?":
"Your ACV is $X,XXX [from claim document]. ACV stands for Actual Cash Value, which means [definition from glossary]..."

Smart Expert Answer:"""
            
            try:
                response = llm.invoke(prompt)
                return {"answer": response.content}
            except Exception as e:
                return {"answer": f"Error generating smart response: {str(e)}"}
        
        # Return smart system (same structure as your original)
        class SmartClaimRAGChain:
            def invoke(self, input_dict):
                return smart_answer(input_dict["input"])
        
        # Clean up memory
        del documents
        gc.collect()
        
        return SmartClaimRAGChain()
        
    except Exception as e:
        print(f"[ERROR] Error in create_smart_claim_system_langchain: {str(e)}")
        # Fallback to regular processing
        return process_pdf_from_file(user_pdf_path, api_key)
