import google.generativeai as genai
import PyPDF2
import re
import gc
import io
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

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

class SimpleVectorStore:
    """Simple vector store using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.vectors = None
        
    def add_documents(self, docs_with_metadata):
        """Add documents with metadata"""
        for doc_text, meta in docs_with_metadata:
            self.documents.append(doc_text)
            self.metadata.append(meta)
        
        # Create TF-IDF vectors
        if self.documents:
            self.vectors = self.vectorizer.fit_transform(self.documents)
            print(f"[DEBUG] Vector store created with {len(self.documents)} documents")
    
    def search(self, query, k=5):
        """Search for similar documents"""
        if not self.vectors:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': similarities[idx]
                })
        
        print(f"[DEBUG] Vector search found {len(results)} relevant documents for query: '{query[:50]}...'")
        return results

def create_glossary_vector_store(glossary_path, api_key):
    """Create vector store from insurance glossary PDF"""
    try:
        print(f"[DEBUG] Creating vector store from: {glossary_path}")
        
        # Extract text from PDF using PyPDF2
        glossary_text = ""
        with open(glossary_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"[DEBUG] PDF loaded: {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    glossary_text += page_text + "\n\n"
                    print(f"[DEBUG] Page {page_num + 1}: {len(page_text)} characters")
        
        if not glossary_text.strip():
            print("[ERROR] No text extracted from PDF")
            return None
        
        # Clean text
        glossary_text = re.sub(r'[\n\xa0\s]+', ' ', glossary_text).strip()
        glossary_text = re.sub(r'\u200b', '', glossary_text).strip()
        
        print(f"[DEBUG] Total glossary text: {len(glossary_text)} characters")
        
        # Split into chunks for better retrieval
        chunks = []
        chunk_size = 600
        overlap = 100
        
        for i in range(0, len(glossary_text), chunk_size - overlap):
            chunk_text = glossary_text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append((
                    chunk_text.strip(),
                    {
                        'source': 'insurance_glossary',
                        'file': glossary_path,
                        'chunk_id': len(chunks)
                    }
                ))
        
        print(f"[DEBUG] Created {len(chunks)} glossary chunks")
        
        # Create vector store
        vector_store = SimpleVectorStore()
        vector_store.add_documents(chunks)
        
        print(f"[SUCCESS] Vector store created with {len(chunks)} glossary chunks")
        return vector_store
        
    except Exception as e:
        print(f"[ERROR] Exception in create_glossary_vector_store: {str(e)}")
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}")
        return None

def create_intelligent_claim_system(user_pdf_path, glossary_vector_store, api_key):
    """Create intelligent system that searches both claim and glossary using vectors"""
    try:
        print(f"[DEBUG] Creating intelligent claim system")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
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
        
        # Create user document chunks for vector search
        user_chunks = []
        chunk_size = 800
        overlap = 100
        
        for i in range(0, len(user_text), chunk_size - overlap):
            chunk_text = user_text[i:i + chunk_size]
            if chunk_text.strip():
                user_chunks.append((
                    chunk_text.strip(),
                    {
                        'source': 'user_claim',
                        'file': user_pdf_path,
                        'chunk_id': len(user_chunks)
                    }
                ))
        
        # Create combined vector store
        combined_vector_store = SimpleVectorStore()
        
        # Add user claim chunks
        combined_vector_store.add_documents(user_chunks)
        
        # Add glossary chunks to the same vector store
        glossary_docs = []
        for doc in glossary_vector_store.documents:
            glossary_docs.append((doc, {'source': 'insurance_glossary'}))
        
        # Combine all documents
        all_documents = user_chunks + glossary_docs
        
        # Create final combined vector store
        final_vector_store = SimpleVectorStore()
        final_vector_store.add_documents(all_documents)
        
        print(f"[DEBUG] Combined vector store created:")
        print(f"  - User claim chunks: {len(user_chunks)}")
        print(f"  - Glossary chunks: {len(glossary_docs)}")
        print(f"  - Total searchable chunks: {len(all_documents)}")
        
        def intelligent_answer(question):
            """Use vector search to find relevant info from both sources"""
            
            # Search for relevant chunks using vector similarity
            relevant_chunks = final_vector_store.search(question, k=8)
            
            if not relevant_chunks:
                return {"answer": "No relevant information found in the documents."}
            
            # Separate claim and glossary information
            claim_info = []
            glossary_info = []
            
            for chunk in relevant_chunks:
                if chunk['metadata']['source'] == 'user_claim':
                    claim_info.append(chunk['content'])
                elif chunk['metadata']['source'] == 'insurance_glossary':
                    glossary_info.append(chunk['content'])
            
            # Build context from most relevant information
            context = ""
            
            if claim_info:
                context += "INFORMATION FROM YOUR CLAIM DOCUMENT:\n"
                context += "\n".join(claim_info[:3])  # Top 3 most relevant
                context += "\n\n"
            
            if glossary_info:
                context += "INSURANCE TERMS DEFINITIONS:\n"
                context += "\n".join(glossary_info[:3])  # Top 3 most relevant
                context += "\n\n"
            
            # Enhanced prompt for intelligent answering
            prompt = f"""You are an expert insurance advisor. Use the information below to answer the user's question.

{context}

Question: {question}

Instructions:
1. First, look for specific information (amounts, dates, details) from the CLAIM DOCUMENT
2. Then, use the INSURANCE TERMS DEFINITIONS to explain terminology in simple language
3. Combine both to give a complete, educational answer

Answer format:
- Start with specific information from the claim (if available)
- Explain any insurance terms in plain English
- Be
