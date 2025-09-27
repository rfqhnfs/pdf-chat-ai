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
    page_title="Insurance Claims AI Assistant", 
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

# INSURANCE GLOSSARY INTEGRATION
@st.cache_resource
def load_insurance_glossary():
    """Load and cache your insurance terms glossary PDF"""
    glossary_path = "insurance_glossary.pdf"  # ← Your glossary PDF goes here
    
    if not os.path.exists(glossary_path):
        st.warning(f"⚠️ Insurance glossary not found: {glossary_path}")
        st.info("Add your insurance terms glossary as 'insurance_glossary.pdf' for enhanced explanations")
        return []
    
    try:
        # Extract the glossary chunks for definitions
        glossary_chunks = functions.get_insurance_glossary_chunks(glossary_path)
        
        st.success(f"✅ Insurance glossary loaded: {len(glossary_chunks)} term definitions available")
        return glossary_chunks
    
    except Exception as e:
        st.error(f"❌ Error loading insurance glossary: {str(e)}")
        return []

# Load insurance glossary once (cached)
glossary_chunks = load_insurance_glossary()

# Enhanced CSS
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
    .glossary-indicator {
        background-color: #e8f5e8;
        border-left: 3px solid #28a745;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .expert-answer {
        background: linear-gradient(145deg, #f0f8ff, #e6f3ff);
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📄 Insurance Claims AI Assistant</h1>", unsafe_allow_html=True)

# Show glossary status
if glossary_chunks:
    st.markdown(f"""
    <div class="glossary-indicator">
        📚 Insurance Expert Mode Active - {len(glossary_chunks)} term definitions loaded for detailed explanations
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
uploaded_file = st.file_uploader("📁 Upload Your Insurance Claim Document (Max: 5MB)", type=['pdf'], key='pdf_upload')

if uploaded_file is not None and check_file_size(uploaded_file):
    # Layout
    container_pdf, container_chat = st.columns([0.45, 0.55], gap='small')
    
    with container_pdf:
        # Process PDF with Insurance Expert Enhancement
        if ss.rag_chain is None:
            with st.spinner("Processing your claim with insurance expert knowledge..."):
                try:
                    # Save and process file
                    with open("temp.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
                    # ENHANCED: Process with insurance glossary integration
                    if glossary_chunks:
                        ss.rag_chain = functions.process_claim_with_glossary(
                            "temp.pdf", 
                            glossary_chunks, 
                            api_key_gemini
                        )
                        success_msg = "✅ Claim processed with insurance expert knowledge!"
                    else:
                        # Fallback to original processing
                        ss.rag_chain = functions.process_pdf_from_file("temp.pdf", api_key_gemini)
                        success_msg = "✅ Claim processed successfully!"
                    
                    ss.pdf_data = uploaded_file.getbuffer()
                    
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                        
                    st.success(success_msg)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                    st.stop()
        
        st.markdown('<p class="section-title">📄 Your Claim Document</p>', unsafe_allow_html=True)
        
        # PDF Viewer
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
                    st.info("📄 Claim document ready for analysis")
        
    with container_chat:
        # Auto-extraction with explanations
        st.markdown('<p class="section-title">🔍 Key Information Extraction</p>', unsafe_allow_html=True)
        
        if ss.auto_extraction_results is None and ss.rag_chain is not None:
            # Enhanced extraction question with explanations
            extraction_question = """Please extract and explain the following information from this insurance claim:

1. RCV (Replacement Cost Value) - What is the RCV amount and what does RCV mean?
2. ACV (Actual Cash Value) - What is the ACV amount and what does ACV mean?  
3. Depreciation - What is the depreciation amount withheld and what does depreciation mean in insurance terms?

For each item, provide both the specific amount from the document AND a brief explanation of what the term means."""
            
            with st.spinner("Extracting and explaining key information..."):
                try:
                    result = ss.rag_chain.invoke({"input": extraction_question})
                    combined_answer = result['answer']
                    
                    # Clean text
                    def clean_text(text):
                        text = re.sub(r'\*+', '', text)
                        text = re.sub(r'\[\$\]', '$', text)
                        text = re.sub(r'[#_`]', '', text)
                        text = re.sub(r'\s+', ' ', text)
                        return text.strip()
                    
                    # Enhanced parsing for explanations
                    lines = combined_answer.split('\n')
                    rcv_answer = "Not found in document"
                    acv_answer = "Not found in document" 
                    dep_answer = "Not found in document"
                    
                    current_section = ""
                    temp_content = []
                    
                    for line in lines:
                        clean_line = clean_text(line)
                        if not clean_line:
                            continue
                            
                        line_lower = clean_line.lower()
                        
                        if 'rcv' in line_lower or 'replacement cost' in line_lower:
                            if temp_content and current_section:
                                # Save previous section
                                content = ' '.join(temp_content)
                                if 'rcv' in current_section:
                                    rcv_answer = content
                                elif 'acv' in current_section:
                                    acv_answer = content
                                elif 'depreciation' in current_section:
                                    dep_answer = content
                            
                            current_section = 'rcv'
                            temp_content = [clean_line]
                        elif 'acv' in line_lower or 'actual cash' in line_lower:
                            if temp_content and current_section:
                                content = ' '.join(temp_content)
                                if 'rcv' in current_section:
                                    rcv_answer = content
                                elif 'acv' in current_section:
                                    acv_answer = content
                                elif 'depreciation' in current_section:
                                    dep_answer = content
                            
                            current_section = 'acv'
                            temp_content = [clean_line]
                        elif 'depreciation' in line_lower:
                            if temp_content and current_section:
                                content = ' '.join(temp_content)
                                if 'rcv' in current_section:
                                    rcv_answer = content
                                elif 'acv' in current_section:
                                    acv_answer = content
                                elif 'depreciation' in current_section:
                                    dep_answer = content
                            
                            current_section = 'depreciation'
                            temp_content = [clean_line]
                        else:
                            if current_section:
                                temp_content.append(clean_line)
                    
                    # Save the last section
                    if temp_content and current_section:
                        content = ' '.join(temp_content)
                        if 'rcv' in current_section:
                            rcv_answer = content
                        elif 'acv' in current_section:
                            acv_answer = content
                        elif 'depreciation' in current_section:
                            dep_answer = content
                    
                    ss.auto_extraction_results = [
                        ("💰 RCV", rcv_answer, "#e3f2fd"),
                        ("💵 ACV", acv_answer, "#f3e5f5"),
                        ("📉 Depreciation", dep_answer, "#fff3e0")
                    ]
                    
                except Exception as e:
                    ss.auto_extraction_results = [
                        ("💰 RCV", "Extraction failed - please ask specific questions", "#e3f2fd"),
                        ("💵 ACV", "Extraction failed - please ask specific questions", "#f3e5f5"),
                        ("📉 Depreciation", "Extraction failed - please ask specific questions", "#fff3e0")
                    ]
        
        # Display enhanced results
        if ss.auto_extraction_results:
            for i, (label, answer, bg_color) in enumerate(ss.auto_extraction_results):
                st.markdown(f"""
                <div class="tall-extraction-card" style="background-color: {bg_color};">
                    <div style="font-weight: bold; font-size: 1rem; color: #1565c0; margin-bottom: 0.5rem;">{label}</div>
                    <div class="value-text" style="font-size: 0.8rem; line-height: 1.3;">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Q&A section with expert mode
        st.markdown('<p class="section-title">💬 Ask Your Insurance Questions</p>', unsafe_allow_html=True)
        
        if glossary_chunks:
            st.info("💡 **Expert Mode**: Ask about any insurance terms for detailed explanations!")
        
        if ss.rag_chain is not None:
            with st.form(key="question_form", clear_on_submit=True):
                user_message = st.text_input("Question:", placeholder="e.g., What is my ACV? How much is my deductible?", label_visibility="collapsed")
                ask_button = st.form_submit_button("🚀 Ask Expert", type="primary")
            
            if ask_button and user_message.strip():
                with st.spinner("Consulting insurance expert..."):
                    try:
                        result = ss.rag_chain.invoke({"input": user_message})
                        
                        def clean_answer(text):
                            text = re.sub(r'\*+', '', text)
                            text = re.sub(r'[#_`]', '', text)
                            text = re.sub(r'\s+', ' ', text)
                            return text.strip()
                        
                        cleaned_answer = clean_answer(result['answer'])
                        
                        st.markdown("**Expert Answer:**")
                        st.markdown(f"""
                        <div class="expert-answer">
                            {cleaned_answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show expert enhancement indicator
                        if glossary_chunks:
                            st.caption("👨‍💼 Answer provided with professional insurance terminology explanations")
                        
                        try:
                            tc = TokenCount()
                            tokens = tc.num_tokens_from_string(str(result))
                            st.caption(f"📊 ~{tokens} tokens")
                        except:
                            pass
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Enhanced quick questions
            st.markdown("**Common Insurance Questions:**")
            quick_questions = ["What's my ACV?", "What's my RCV?", "How much depreciation?", "What's my deductible?"]
            
            quick_cols = st.columns(2)
            for i, quick_q in enumerate(quick_questions):
                with quick_cols[i % 2]:
                    if st.button(quick_q, key=f"quick_{i}"):
                        full_questions = [
                            "What is my ACV amount and what does ACV mean?",
                            "What is my RCV amount and what does RCV mean?",
                            "What is the depreciation amount and what does depreciation mean in insurance?", 
                            "What is my deductible amount and what does deductible mean?"
                        ]
                        
                        with st.spinner("Getting expert explanation..."):
                            try:
                                result = ss.rag_chain.invoke({"input": full_questions[i]})
                                
                                def clean_answer(text):
                                    text = re.sub(r'\*+', '', text)
                                    text = re.sub(r'[#_`]', '', text)
                                    text = re.sub(r'\s+', ' ', text)
                                    return text.strip()
                                
                                cleaned_answer = clean_answer(result['answer'])
                                
                                st.markdown(f"""
                                <div class="expert-answer">
                                    <strong>{quick_q}</strong><br><br>
                                    {cleaned_answer}
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
        
        # Clear memory button
        if st.button("🧹 Clear & Upload New Claim"):
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
        <h3>👋 Upload Your Insurance Claim Document</h3>
        <p>Get expert explanations of insurance terms and amounts</p>
        <p><strong>📏 Maximum file size: 5MB</strong></p>
        <p style="font-size: 0.9rem; color: #666;">💡 The AI will explain insurance terms in simple language</p>
    </div>
    """, unsafe_allow_html=True)
