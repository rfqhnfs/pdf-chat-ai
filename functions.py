import google.generativeai as genai
import PyPDF2
import re
import gc
import os

def process_pdf_from_file(file_path, api_key):
    """
    Process PDF using google-generativeai 0.7.2 - optimized for your exact setup
    """
    try:
        # Configure Gemini with your version
        genai.configure(api_key=api_key)
        
        # Try different model names for your version
        model_names = [
            "gemini-pro",
            "models/gemini-pro", 
            "gemini-1.5-flash",
            "models/gemini-1.5-flash"
        ]
        
        model = None
        working_model_name = None
        
        # Find working model
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model with a simple prompt
                test_response = model.generate_content("Hello")
                working_model_name = model_name
                print(f"[SUCCESS] Working model found: {model_name}")
                break
            except Exception as e:
                print(f"[DEBUG] Model {model_name} failed: {str(e)}")
                continue
        
        if not model:
            raise Exception("No working model found for your google-generativeai version")
        
        # Extract text from PDF using PyPDF2
        pdf_text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + " "
        
        # Clean text
        pdf_text = re.sub(r'[\n\xa0\s]+', ' ', pdf_text).strip()
        pdf_text = re.sub(r'\u200b', '', pdf_text).strip()
        
        # Limit text for memory efficiency
        if len(pdf_text) > 12000:
            pdf_text = pdf_text[:12000]
        
        print(f"[DEBUG] Using model: {working_model_name} | PDF text length: {len(pdf_text)}")
        
        def answer_question(question):
            prompt = f"""You are an AI assistant that answers questions based on document content.

Document Content:
{pdf_text}

Question: {question}

Instructions:
- Answer based ONLY on the document content above
- Be accurate and concise  
- If information is not in the document, say "Information not found in the document"
- For insurance documents, look for RCV, ACV, depreciation amounts, claim details, etc.

Answer:"""
            
            try:
                response = model.generate_content(prompt)
                return {"answer": response.text}
            except Exception as e:
                return {"answer": f"Error generating response: {str(e)}"}
        
        # Clean up memory
        gc.collect()
        
        # Return chain-like object
        class OptimizedRAG:
            def invoke(self, input_dict):
                return answer_question(input_dict["input"])
        
        return OptimizedRAG()
        
    except Exception as e:
        gc.collect()
        raise Exception(f"PDF processing failed: {str(e)}")

def extract_glossary_text(glossary_path):
    """Extract text from insurance glossary PDF using PyPDF2"""
    try:
        print(f"[DEBUG] Extracting glossary text: {glossary_path}")
        
        # Check if file exists
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
        print(f"[ERROR] Exception in extract_glossary_text: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return None

def create_expert_claim_system(user_pdf_path, glossary_text, api_key):
    """Create expert system that combines claim document with glossary"""
    try:
        print(f"[DEBUG] Creating expert claim system")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Find working model (same logic as above)
        model_names = [
            "gemini-pro",
            "models/gemini-pro", 
            "gemini-1.5-flash",
            "models/gemini-1.5-flash"
        ]
        
        model = None
        working_model_name = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model
                test_response = model.generate_content("Hello")
                working_model_name = model_name
                print(f"[SUCCESS] Expert system using model: {model_name}")
                break
            except Exception as e:
                print(f"[DEBUG] Model {model_name} failed: {str(e)}")
                continue
        
        if not model:
            raise Exception("No working model found for expert system")
        
        # Extract text from user's claim document using PyPDF2
        user_text = ""
        with open(user_pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    user_text += page_text + " "
        
        # Clean user text
        user_text = re.sub(r'[\n\xa0\s]+', ' ', user_text).strip()
        user_text = re.sub(r'\u200b', '', user_text).strip()
        
        # Limit text for memory efficiency
        if len(user_text) > 10000:
            user_text = user_text[:10000]
        
        # Limit glossary text
        if len(glossary_text) > 5000:
            glossary_text = glossary_text[:5000]
        
        print(f"[DEBUG] Expert system content - User claim: {len(user_text)} chars, Glossary: {len(glossary_text)} chars")
        
        def expert_answer(question):
            """Provide expert answers using both user document and glossary"""
            
            # Create enhanced prompt that uses both documents
            prompt = f"""You are an expert insurance advisor helping customers understand their insurance claims.

USER'S INSURANCE CLAIM DOCUMENT:
{user_text}

INSURANCE TERMS GLOSSARY (Contains definitions for RCV, ACV, Depreciation, Deductible, etc.):
{glossary_text}

Question: {question}

Instructions:
1. **Find specific information** from the USER'S CLAIM DOCUMENT (amounts, dates, policy details, etc.)
2. **Use the GLOSSARY** to explain any insurance terminology in simple, clear language
3. **Combine both** to give a complete, educational answer

Your answer should:
- Start with specific information from the claim document (if available)
- Explain insurance terms using the glossary definitions in simple language
- Be educational and easy to understand for non-insurance experts
- If information isn't available, clearly state that

Example format for "What is my ACV?":
"Your ACV is $X,XXX [from claim document]. ACV stands for Actual Cash Value, which means [definition from glossary in simple terms]..."

Expert Answer:"""
            
            try:
                response = model.generate_content(prompt)
                return {"answer": response.text}
            except Exception as e:
                return {"answer": f"Error generating expert response: {str(e)}"}
        
        # Return expert system
        class ExpertClaimRAG:
            def invoke(self, input_dict):
                return expert_answer(input_dict["input"])
        
        gc.collect()  # Clean up memory
        return ExpertClaimRAG()
        
    except Exception as e:
        print(f"[ERROR] Error in create_expert_claim_system: {str(e)}")
        # Fallback to regular processing
        return process_pdf_from_file(user_pdf_path, api_key)
