import google.generativeai as genai
import PyPDF2
import re
import gc
import io

def process_pdf_from_file(file_path, api_key):
    """
    Pure Python PDF processing - no LangChain dependencies
    100% compatible with Streamlit Cloud
    """
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
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
        if len(pdf_text) > 15000:
            pdf_text = pdf_text[:15000]
        
        def answer_question(question):
            prompt = f"""
You are an AI assistant that answers questions based on document content.

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
        class PurePythonRAG:
            def invoke(self, input_dict):
                return answer_question(input_dict["input"])
        
        return PurePythonRAG()
        
    except Exception as e:
        gc.collect()
        raise Exception(f"PDF processing failed: {str(e)}")

def get_pdf_chunks(pdf_path):
    """Extract document chunks from a PDF file for knowledge base integration"""
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split into chunks (same settings as your main processing)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to identify this as knowledge base content
        for chunk in chunks:
            chunk.metadata['source_type'] = 'knowledge_base'
            chunk.metadata['source_file'] = pdf_path
        
        return chunks
        
    except Exception as e:
        print(f"Error processing knowledge base PDF: {str(e)}")
        return []

def get_insurance_glossary_chunks(glossary_path):
    """Extract insurance term definitions for expert explanations"""
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Load the glossary PDF
        loader = PyPDFLoader(glossary_path)
        documents = loader.load()
        
        # Split into smaller chunks for better term matching
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for definitions
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to identify this as glossary content
        for chunk in chunks:
            chunk.metadata['source_type'] = 'insurance_glossary'
            chunk.metadata['source_file'] = glossary_path
            chunk.metadata['content_purpose'] = 'term_definitions'
        
        return chunks
        
    except Exception as e:
        print(f"Error processing insurance glossary: {str(e)}")
        return []

def process_claim_with_glossary(user_pdf_path, glossary_chunks, api_key):
    """Process insurance claim with expert glossary explanations"""
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate
        
        # Process user's claim document
        loader = PyPDFLoader(user_pdf_path)
        user_documents = loader.load()
        
        # Add metadata to user documents
        for doc in user_documents:
            doc.metadata['source_type'] = 'user_claim'
            doc.metadata['source_file'] = user_pdf_path
            doc.metadata['content_purpose'] = 'claim_data'
        
        # Split user documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        user_chunks = text_splitter.split_documents(user_documents)
        
        # COMBINE: User claim chunks + Insurance glossary chunks
        all_chunks = user_chunks + glossary_chunks
        print(f"Insurance Expert Setup: {len(all_chunks)} total chunks")
        print(f"  - Claim document: {len(user_chunks)} chunks")
        print(f"  - Insurance glossary: {len(glossary_chunks)} term definitions")
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Create combined vector store
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Get more chunks to include both claim data and definitions
        )
        
        # Set up the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.2  # Lower temperature for more consistent explanations
        )
        
        # EXPERT INSURANCE ADVISOR PROMPT
        system_prompt = """You are an expert insurance advisor helping customers understand their insurance claims. 
        You have access to both their specific claim document and a comprehensive insurance terms glossary.

        When answering questions:

        1. **Find the specific information** from their claim document (amounts, dates, details)
        2. **Explain insurance terms** using the glossary definitions in simple language
        3. **Combine both** to give complete, educational answers

        Example format for "What is my ACV?":
        "Your ACV (Actual Cash Value) is $1,000. ACV stands for Actual Cash Value, which means the current worth of your property after accounting for age, wear, and depreciation - essentially what it would cost to replace your item with a similar used one in today's market."

        Always:
        - Answer the specific question with actual numbers/details from their document
        - Explain what insurance terms mean in plain English  
        - Be helpful and educational
        - If information isn't in the documents, say so clearly

        Context: {context}

        Question: {input}

        Expert Answer: """
        
        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Create the expert chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        print(f"Error in process_claim_with_glossary: {str(e)}")
        # Fallback to regular processing
        return process_pdf_from_file(user_pdf_path, api_key)


def get_insurance_glossary_chunks(glossary_path):
    """Extract insurance term definitions for expert explanations"""
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        print(f"[DEBUG] Starting to process: {glossary_path}")
        
        # Check if file exists first
        if not os.path.exists(glossary_path):
            print(f"[ERROR] File does not exist: {glossary_path}")
            return []
        
        # Check file size
        file_size = os.path.getsize(glossary_path)
        print(f"[DEBUG] File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"[ERROR] File is empty: {glossary_path}")
            return []
        
        # Load the glossary PDF
        loader = PyPDFLoader(glossary_path)
        print("[DEBUG] PyPDFLoader created")
        
        documents = loader.load()
        print(f"[DEBUG] PDF loaded: {len(documents)} pages")
        
        if not documents:
            print("[ERROR] No documents loaded from PDF")
            return []
        
        # Print first page content for debugging
        if len(documents) > 0:
            first_page = documents[0].page_content[:200]
            print(f"[DEBUG] First page preview: {first_page}...")
        
        # Split into smaller chunks for better term matching
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks for definitions
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"[DEBUG] Created {len(chunks)} chunks")
        
        # Add metadata to identify this as glossary content
        for i, chunk in enumerate(chunks):
            chunk.metadata['source_type'] = 'insurance_glossary'
            chunk.metadata['source_file'] = glossary_path
            chunk.metadata['content_purpose'] = 'term_definitions'
            chunk.metadata['chunk_id'] = i
        
        print(f"[SUCCESS] Processing complete: {len(chunks)} chunks ready")
        return chunks
        
    except Exception as e:
        print(f"[ERROR] Exception in get_insurance_glossary_chunks: {str(e)}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return []


