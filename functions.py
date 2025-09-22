from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re
import gc

def process_pdf_from_file(file_path, api_key):
    """
    Optimized PDF processing for Streamlit Cloud with memory management
    """
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Clean text
        for i in range(len(documents)):
            documents[i].page_content = re.sub(r'[\n\xa0\s]+', ' ', documents[i].page_content).strip()
            documents[i].page_content = re.sub(r'\u200b', '', documents[i].page_content).strip()
        
        # Optimized text splitting for memory efficiency
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,    # Reduced for memory optimization
            chunk_overlap=100  # Reduced overlap
        )
        splits = text_splitter.split_documents(documents)
        
        # Limit number of chunks for memory
        if len(splits) > 50:
            splits = splits[:50]  # Keep only first 50 chunks
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/gemini-embedding-001"
        )
        
        # Create vector store
        db = FAISS.from_documents(splits, embeddings)
        
        # Optimized retriever for Streamlit Cloud
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,        # Reduced from 12
                "fetch_k": 20, # Reduced from 30  
                "lambda_mult": 0.5
            }
        )

        # LLM with conservative settings
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",  # Use latest efficient model
            temperature=0,
            max_tokens=250,  # Reduced for faster processing
            google_api_key=api_key
        )

        # Optimized prompt
        system_prompt = (
            "You are an AI assistant that answers questions based on document content. "
            "Use the provided context to give accurate and concise answers. "
            "Look through the context carefully as information may be scattered. "
            "If you cannot find the answer in the provided context, clearly state that."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Clean up memory
        del documents
        del splits
        gc.collect()
        
        return rag_chain
        
    except Exception as e:
        # Clean up on error
        gc.collect()
        raise e
