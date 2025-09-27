import streamlit as st
from streamlit import session_state as ss
import functions
from token_count import TokenCount
import re
import base64
import os
import gc

# Page config MUST be first
st.set_page_config(
    page_title="PDF Chat AI", 
    page_icon="📄", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Get API key
try:
    api_key_gemini = st.secrets["GEMINI_API"]
except (KeyError, FileNotFoundError):
    st.error("🚨 GEMINI_API key not found in secrets.")
    st.info("Please add GEMINI_API in app settings → Secrets")
    st.stop()

# KNOWLEDGE BASE INTEGRATION - Add this section
@st.cache_resource
def load_knowledge_base():
    """Load and cache your additional PDF knowledge base"""
    knowledge_base_path = "knowledge_base.pdf"  # ← Your additional PDF goes here
    
    if not os.path.exists(knowledge_base_path):
        st.warning(f"⚠️ Knowledge base file not found: {knowledge_base_path}")
        return None, []
    
    try:
        # Extract the processed chunks for later combination
        knowledge_chunks = functions.get_pdf_chunks(knowledge_base_path)
        
        st.success(f"✅ Knowledge base loaded: {knowledge_base_path}")
        return None, knowledge_chunks
    
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        return None, []

# Load knowledge base once (cached)
knowledge_rag_chain, knowledge_chunks = load_knowledge_base()

# Minimal CSS
st.markdown("""
<style>
    .main-header {
        text-align: center; 
        color: #1f77b4; 
        font-size: 1.8rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .tall-extraction-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-height: 120px;
    }
    .value-text {
        color: #1565c0;
        font-weight: bold;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        line-height: 1.4;
        word-wrap: break-word;
    }
    .section-title {
        font-size: 1.2rem;
        color: #1f77b4;
        margin: 0.5rem 0;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.3rem;
    }
    .answer-compact {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .block-container {padding-top: 1rem !important;}
    .knowledge-indicator {
        background-color: #e8f5e8;
        border-left: 3px solid #28a745;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📄 PDF Chat AI</h1>", unsafe_allow_html=True)

# Show knowledge base status
if knowledge_chunks:
    st.markdown(f"""
    <div class="knowledge-indicator">
        🧠 Enhanced with knowledge base ({len(knowledge_chunks)} additional knowledge chunks loaded)
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'rag_chain' not in ss:
    ss.rag_chain = None
if 'auto_extraction_results' not in ss:
    ss.auto_extraction_results = None
if 'pdf_data' not in ss:
    ss.pdf_data = None

# Memory cleanup
def cleanup_memory():
    gc.collect()

# Check file size function
def check_file_size(uploaded_file):
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        if file_size_mb > 5:
            st.error(f"🚨 File too large: {file_size_mb:.1f}MB. Maximum allowed: 5MB")
            return False
        else:
            st.success(f"✅ File size: {file_size_mb:.1f}MB")
            return True
    return False

# File uploader
uploaded_file = st.file_uploader("📁 Choose PDF (Max: 5MB)", type=['pdf'], key='pdf_upload')

if uploaded_file is not None and check_file_size(uploaded_file):
    # Layout
    container_pdf, container_chat = st.columns([0.45, 0.55], gap='small')
    
    with container_pdf:
        # Process PDF once - NOW WITH KNOWLEDGE BASE INTEGRATION
        if ss.rag_chain is None:
            with st.spinner("Processing PDF with knowledge base..."):
                try:
                    # Save and process file
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
                    # ENHANCED: Process with knowledge base integration
                    if knowledge_chunks:
                        ss.rag_chain = functions.process_pdf_with_knowledge_base(
                            "temp.pdf", 
                            knowledge_chunks, 
                            api_key_gemini
                        )
                    else:
                        # Fallback to original processing if no knowledge base
                        ss.rag_chain = functions.process_pdf_from_file("temp.pdf", api_key_gemini)
                    
                    ss.pdf_data = uploaded_file.getbuffer()
                    
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                        
                    if knowledge_chunks:
                        st.success("✅ PDF processed with knowledge base enhancement!")
                    else:
                        st.success("✅ PDF processed successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                    st.stop()
        
        st.markdown('<p class="section-title">📄 PDF Document</p>', unsafe_allow_html=True)
        
        # **MINIMAL PDF VIEWER** - No extra boxes or download buttons
        if ss.pdf_data is not None:
            try:
                base64_pdf = base64.b64encode(ss.pdf_data).decode('utf-8')
                st.markdown(f"""
                <iframe 
                    src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" 
                    height="650" 
                    style="border: 1px solid #ddd; border-radius: 8px;">
                </iframe>
                """, unsafe_allow_html=True)
            except:
                try:
                    st.markdown(f"""
                    <embed 
                        src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" 
                        height="650" 
                        type="application/pdf"
                        style="border: 1px solid #ddd; border-radius: 8px;">
                    """, unsafe_allow_html=True)
                except:
                    st.info("📄 PDF ready for analysis")
        
    with container_chat:
        # Auto-extraction
        st.markdown('<p class="section-title">🔍 Auto Extraction</p>', unsafe_allow_html=True)
        
        if ss.auto_extraction_results is None and ss.rag_chain is not None:
            combined_question = """Please extract the following information from this insurance document:
1. RCV (Replacement Cost Value) - How much is the RCV?
2. ACV (Actual Cash Value) - How much is the ACV? 
3. Depreciation Withheld - How much is the depreciation withheld amount?

Format your answer clearly with labels for each value."""
            
            with st.spinner("Extracting key info..."):
                try:
                    result = ss.rag_chain.invoke({"input": combined_question})
                    combined_answer = result['answer']
                    
                    # Clean text
                    def clean_text(text):
                        text = re.sub(r'\*+', '', text)
                        text = re.sub(r'\[\$\]', '$', text)
                        text = re.sub(r'[#_`]', '', text)
                        text = re.sub(r'\s+', ' ', text)
                        return text.strip()
                    
                    # Parse answer
                    lines = combined_answer.split('\n')
                    rcv_answer = "Not found"
                    acv_answer = "Not found" 
                    dep_answer = "Not found"
                    
                    for line in lines:
                        clean_line = clean_text(line)
                        line_lower = clean_line.lower()
                        
                        if 'rcv' in line_lower or 'replacement cost' in line_lower:
                            rcv_answer = clean_line
                        elif 'acv' in line_lower or 'actual cash' in line_lower:
                            acv_answer = clean_line
                        elif 'depreciation' in line_lower:
                            dep_answer = clean_line
                    
                    if rcv_answer == "Not found" and acv_answer == "Not found":
                        cleaned_full = clean_text(combined_answer)
                        rcv_answer = acv_answer = dep_answer = cleaned_full[:200] + "..."
                    
                    ss.auto_extraction_results = [
                        ("💰 RCV", rcv_answer, "#e3f2fd"),
                        ("💵 ACV", acv_answer, "#f3e5f5"),
                        ("📉 Depreciation", dep_answer, "#fff3e0")
                    ]
                    
                except Exception as e:
                    ss.auto_extraction_results = [
                        ("💰 RCV", "Extraction failed", "#e3f2fd"),
                        ("💵 ACV", "Extraction failed", "#f3e5f5"),
                        ("📉 Depreciation", "Extraction failed", "#fff3e0")
                    ]
        
        # Display results
        if ss.auto_extraction_results:
            extraction_cols = st.columns(3)
            for i, (label, answer, bg_color) in enumerate(ss.auto_extraction_results):
                with extraction_cols[i]:
                    st.markdown(f"""
                    <div class="tall-extraction-card" style="background-color: {bg_color};">
                        <div style="font-weight: bold; font-size: 0.95rem; color: #1565c0;">{label}</div>
                        <div class="value-text">{answer}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Q&A section
        st.markdown('<p class="section-title">💬 Ask Questions</p>', unsafe_allow_html=True)
        
        if ss.rag_chain is not None:
            with st.form(key="question_form", clear_on_submit=True):
                user_message = st.text_input("Question:", placeholder="Ask anything...", label_visibility="collapsed")
                ask_button = st.form_submit_button("🚀 Ask", type="primary")
            
            if ask_button and user_message.strip():
                with st.spinner("Thinking..."):
                    try:
                        result = ss.rag_chain.invoke({"input": user_message})
                        
                        def clean_answer(text):
                            text = re.sub(r'\*+', '', text)
                            text = re.sub(r'[#_`]', '', text)
                            text = re.sub(r'\s+', ' ', text)
                            return text.strip()
                        
                        cleaned_answer = clean_answer(result['answer'])
                        
                        st.markdown("**Answer:**")
                        st.markdown(f"""
                        <div class="answer-compact">
                            {cleaned_answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show knowledge enhancement indicator
                        if knowledge_chunks:
                            st.caption("🧠 Answer enhanced with additional knowledge base")
                        
                        try:
                            tc = TokenCount()
                            tokens = tc.num_tokens_from_string(str(result))
                            st.caption(f"📊 ~{tokens} tokens")
                        except:
                            pass
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Quick questions
            st.markdown("**Quick Questions:**")
            quick_questions = ["Total claim?", "Policyholder?", "Damage type?", "When occurred?"]
            
            quick_cols = st.columns(2)
            for i, quick_q in enumerate(quick_questions):
                with quick_cols[i % 2]:
                    if st.button(quick_q, key=f"quick_{i}"):
                        full_questions = [
                            "What is the total claim amount?",
                            "Who is the policyholder?",
                            "What type of damage occurred?", 
                            "When did the incident happen?"
                        ]
                        
                        with st.spinner("Processing..."):
                            try:
                                result = ss.rag_chain.invoke({"input": full_questions[i]})
                                
                                def clean_answer(text):
                                    text = re.sub(r'\*+', '', text)
                                    text = re.sub(r'[#_`]', '', text)
                                    text = re.sub(r'\s+', ' ', text)
                                    return text.strip()
                                
                                cleaned_answer = clean_answer(result['answer'])
                                
                                st.markdown(f"""
                                <div class="answer-compact">
                                    <strong>{quick_q}</strong><br>
                                    {cleaned_answer}
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        # Clear memory button
        if st.button("🧹 Clear & Upload New PDF"):
            cleanup_memory()
            for key in ['rag_chain', 'auto_extraction_results', 'pdf_data']:
                if key in ss:
                    del ss[key]
            st.success("✅ Memory cleared!")
            st.rerun()

# Reset state
elif uploaded_file is None:
    if ss.rag_chain is not None:
        ss.rag_chain = None
        ss.auto_extraction_results = None
        ss.pdf_data = None
        cleanup_memory()
    
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem;">
        <h3>👋 Upload a PDF to get started</h3>
        <p>Automatically extract RCV, ACV, and depreciation amounts</p>
        <p><strong>📏 Maximum file size: 5MB</strong></p>
    </div>
    """, unsafe_allow_html=True)
