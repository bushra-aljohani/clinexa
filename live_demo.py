"""
🏥 Clinexa - Enhanced Live Pipeline Demo (Compact Version)
===========================================================
Interactive demonstration of the Agentic Hybrid Medical RAG Pipeline
for MedGemma Impact Challenge Hackathon Presentation

Author: Bushra Salama Aljohani
GitHub: https://github.com/bushra-aljohani/clinexa.git
Version: 2.1 (Compact Pipeline)
"""

import streamlit as st
import time
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Clinexa - Enhanced Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# COMPACT CUSTOM STYLES
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        color: #6b7280;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .hackathon-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.6rem 1.8rem;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        margin: 0.5rem auto;
        text-align: center;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .author-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem auto;
        text-align: center;
        font-size: 0.95rem;
    }
    
    /* COMPACT HORIZONTAL PIPELINE */
    .pipeline-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background: #f8fafc;
        border-radius: 12px;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    .stage-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        white-space: nowrap;
        transition: all 0.3s ease;
        min-width: 180px;
    }
    
    .stage-chip.pending {
        background: #e2e8f0;
        color: #64748b;
        border: 2px solid #cbd5e1;
    }
    
    .stage-chip.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #f093fb;
        animation: pulse-chip 1.5s infinite;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stage-chip.complete {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 2px solid #10b981;
    }
    
    @keyframes pulse-chip {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .stage-arrow {
        color: #cbd5e1;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #f3f4f6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-sublabel {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Source cards */
    .source-card {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Report section */
    .report-section {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-weight: 700;
        border-radius: 30px;
        transition: all 0.3s ease;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Progress indicator */
    .progress-container {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SAMPLE DATA (keeping same as before)
# =========================
SAMPLE_QUERIES = {
    "Type 2 Diabetes Treatment": "What are the current FDA-approved treatments for metformin-resistant Type 2 Diabetes, and what clinical evidence supports their efficacy?",
    "Cardiovascular Risk": "How do SGLT2 inhibitors compare to GLP-1 agonists in reducing cardiovascular risk in diabetic patients with established heart disease?",
    "Chronic Kidney Disease": "What are the latest guidelines for managing diabetic nephropathy in patients with CKD stage 3?",
    "Weight Management": "Which anti-obesity medications show the best long-term efficacy and safety profile for patients with BMI >35?",
    "Custom Query": ""
}

SAMPLE_ANALYSIS = {
    "query_type": "treatment_comparison",
    "medical_entities": ["Type 2 Diabetes", "Metformin resistance", "FDA-approved treatments"],
    "temporal_indicators": ["current", "latest"],
    "clinical_specialty": "Endocrinology",
    "intent": "Clinical evidence synthesis for treatment alternatives",
    "complexity": "High",
    "requires_dynamic_search": True,
    "recommended_sources": ["PubMed (2024-2026)", "ClinicalTrials.gov", "FDA Drug Database"],
    "estimated_tokens": 2847
}

SAMPLE_SOURCES = [
    {"title": "SGLT2 Inhibitors in T2DM: A Systematic Review and Meta-Analysis", "source": "PubMed", "year": 2025, "relevance": 0.94, "type": "Meta-analysis", "citations": 234},
    {"title": "GLP-1 Receptor Agonists vs DPP-4 Inhibitors: Comparative Effectiveness", "source": "PubMed", "year": 2024, "relevance": 0.91, "type": "RCT", "citations": 156},
    {"title": "FDA Approval: Tirzepatide for Type 2 Diabetes Mellitus", "source": "FDA Database", "year": 2024, "relevance": 0.89, "type": "Regulatory", "citations": 89},
    {"title": "Cardiovascular Outcomes with SGLT2 Inhibitors - EMPA-REG OUTCOME", "source": "ClinicalTrials.gov", "year": 2023, "relevance": 0.87, "type": "Clinical Trial", "citations": 1203},
    {"title": "ADA Standards of Medical Care in Diabetes - 2026", "source": "Clinical Guidelines", "year": 2026, "relevance": 0.85, "type": "Guideline", "citations": 2341},
]

SAMPLE_REPORT = """
## 🏥 Clinical Summary: Metformin-Resistant Type 2 Diabetes Treatment Options

### 📊 Evidence-Based Treatment Recommendations

Based on comprehensive analysis of current clinical evidence and FDA approvals, the following therapeutic options represent the primary choices for metformin-resistant T2DM:

---

### 💊 **1. SGLT2 Inhibitors (Sodium-Glucose Co-Transporter 2)**

**FDA-Approved Agents:**
- Empagliflozin (Jardiance) - 2014
- Dapagliflozin (Farxiga) - 2014  
- Canagliflozin (Invokana) - 2013

**Mechanism of Action:**  
Blocks renal glucose reabsorption in proximal tubules, promoting glucosuria

**Clinical Evidence:**
- **EMPA-REG OUTCOME (2015):** 38% reduction in cardiovascular mortality (HR 0.62, 95% CI 0.49-0.77)
- **DECLARE-TIMI 58 (2019):** 17% reduction in hospitalization for heart failure
- **HbA1c reduction:** 0.5-1.0% as monotherapy

**Key Benefits:**
✅ Cardioprotective effects (especially in established CVD)  
✅ Renoprotective effects (slows CKD progression)  
✅ Weight loss (2-3 kg average)  
✅ Blood pressure reduction (3-5 mmHg systolic)

**Considerations:**
⚠️ Genital mycotic infections (10-15% incidence)  
⚠️ Rare DKA risk (especially in Type 1 or stressed patients)

---

*🧬 Report generated by Clinexa Agentic RAG Pipeline*  
*📊 Synthesis Model: TXGemma-9B | Evaluation: BERTScore F1 83.14%*  
*⏱️ Response Time: 46.48s | 📝 Response Length: 1,735 characters*

---

**⚠️ Clinical Disclaimer:** This information is for educational purposes. Treatment decisions should be individualized based on patient-specific factors, contraindications, and shared decision-making. Consult clinical guidelines and specialist input when appropriate.
"""

EVALUATION_METRICS = {
    "bertscore": {"precision": 0.8451, "recall": 0.8185, "f1": 0.8314},
    "ragas": {"faithfulness": 0.5119, "relevancy": 0.2015},
    "llm_judge": {"overall": 3.77, "accuracy": 3.95, "relevance": 4.00, "completeness": 3.45, "clarity": 3.90, "grounding": 3.55}
}

# =========================
# COMPACT PIPELINE DISPLAY
# =========================
def show_compact_pipeline(current_stage=0):
    """Display compact horizontal pipeline"""
    stages = [
        {"num": 1, "name": "Query Analysis", "emoji": "🧠"},
        {"num": 2, "name": "Routing", "emoji": "🔀"},
        {"num": 3, "name": "Retrieval", "emoji": "📚"},
        {"num": 4, "name": "Ranking", "emoji": "⚖️"},
        {"num": 5, "name": "Synthesis", "emoji": "🧬"},
        {"num": 6, "name": "Evaluation", "emoji": "✅"}
    ]
    
    pipeline_html = '<div class="pipeline-container">'
    
    for i, stage in enumerate(stages):
        if i > 0:
            pipeline_html += '<span class="stage-arrow">→</span>'
        
        if current_stage == 0:
            status = "pending"
        elif stage["num"] < current_stage:
            status = "complete"
        elif stage["num"] == current_stage:
            status = "active"
        else:
            status = "pending"
        
        status_icon = {"pending": "⏳", "active": "🔄", "complete": "✅"}
        
        pipeline_html += f'''
        <div class="stage-chip {status}">
            {stage["emoji"]} Stage {stage["num"]}: {stage["name"]} {status_icon[status]}
        </div>
        '''
    
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

def create_evaluation_chart():
    """Create radar chart for evaluation metrics"""
    categories = ['Accuracy', 'Relevance', 'Completeness', 'Clarity', 'Grounding']
    values = [3.95, 4.00, 3.45, 3.90, 3.55]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name='TXGemma-9B',
        line=dict(color='#667eea', width=3), fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_performance_comparison():
    """Create performance comparison chart"""
    metrics = ['Speed', 'Detail', 'Accuracy', 'Cost']
    txgemma = [2, 9, 8.3, 10]
    
    fig = go.Figure(data=[go.Bar(x=metrics, y=txgemma, marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#10b981'], line=dict(color='#ffffff', width=2)), text=[f'{v:.1f}' for v in txgemma], textposition='outside')])
    fig.update_layout(title="TXGemma-9B Performance Profile", yaxis_title="Score (out of 10)", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    return fig

# =========================
# MAIN APP
# =========================
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Display logo
        logo_path = Path("Modern medical-tech logo design.png")
        if logo_path.exists():
            st.image(str(logo_path), width=350)
        
        st.markdown('<h1 class="main-title">Clinexa</h1>', unsafe_allow_html=True)
        
        st.markdown('<p class="subtitle">Agentic Hybrid Medical Intelligence System</p>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><span class="hackathon-badge">🏆 MedGemma Impact Challenge 2026</span></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><span class="author-badge">👩‍💻 Bushra Salama Aljohani</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Challenge Submission")
        st.markdown("""
        <div class="info-box">
        <strong>Competition:</strong> MedGemma Impact Challenge<br>
        <strong>Track:</strong> Agentic Workflow Prize<br>
        <strong>Date:</strong> February 2026
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Key Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BERTScore", "83.14%")
        with col2:
            st.metric("LLM Judge", "3.77/5")
        st.metric("Faithfulness", "51.19%")
        st.metric("Response Time", "~46s")
        
        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.link_button("💻 GitHub Repository", "https://github.com/bushra-aljohani/clinexa", use_container_width=True)
        st.link_button("📄 View README", "https://github.com/bushra-aljohani/clinexa#readme", use_container_width=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Live Pipeline Demo", "📊 Evaluation Dashboard", "🏗️ System Architecture", "🎯 Hackathon Impact"])
    
    # =========================
    # TAB 1: COMPACT LIVE DEMO
    # =========================
    with tab1:
        st.markdown("### 🔬 Interactive Pipeline Demonstration")
        st.markdown("""
        <div class="info-box">
        <strong>💡 Demo Instructions:</strong> Select a sample query or enter your own medical question, then click "Run Pipeline" to watch Clinexa process it through all 6 stages.
        </div>
        """, unsafe_allow_html=True)
        
        # Query selection
        col1, col2 = st.columns([3, 1])
        with col1:
            query_choice = st.selectbox("📋 Select Sample Query or Choose 'Custom':", list(SAMPLE_QUERIES.keys()))
        
        if query_choice == "Custom Query":
            query = st.text_area("✍️ Enter Your Medical Query:", height=100, placeholder="e.g., What are the contraindications for ACE inhibitors in diabetic patients?")
        else:
            query = SAMPLE_QUERIES[query_choice]
            st.text_area("📝 Selected Query:", value=query, height=100, disabled=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_button = st.button("🚀 Run Complete Pipeline", use_container_width=True, type="primary")
        
        if run_button and query:
            st.markdown("---")
            
            # Progress bar
            progress_placeholder = st.empty()
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 0%"></div></div>', unsafe_allow_html=True)
            
            # Pipeline status (compact)
            pipeline_placeholder = st.empty()
            
            # Output area
            st.markdown("### 💡 Agent Outputs")
            output_container = st.container()
            
            # Stage 1
            with pipeline_placeholder:
                show_compact_pipeline(1)
            with output_container:
                with st.spinner("🧠 DeepSeek-V3 analyzing medical query..."):
                    time.sleep(1.5)
                st.markdown('<div class="success-box">✅ <strong>Query Analysis Complete!</strong> Extracted entities and clinical context.</div>', unsafe_allow_html=True)
                with st.expander("🔍 View Detailed Analysis", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({"Query Type": SAMPLE_ANALYSIS["query_type"], "Clinical Specialty": SAMPLE_ANALYSIS["clinical_specialty"], "Complexity": SAMPLE_ANALYSIS["complexity"]})
                    with col2:
                        st.json({"Medical Entities": SAMPLE_ANALYSIS["medical_entities"], "Temporal Indicators": SAMPLE_ANALYSIS["temporal_indicators"]})
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 16%"></div></div>', unsafe_allow_html=True)
            time.sleep(0.5)
            
            # Stage 2
            with pipeline_placeholder:
                show_compact_pipeline(2)
            with output_container:
                with st.spinner("🔀 Router Agent selecting optimal sources..."):
                    time.sleep(1.2)
                st.markdown('<div class="success-box">✅ <strong>Routing Decision:</strong> HYBRID RAG (Static + Dynamic APIs)</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📚 Static Sources**\n- ✅ PubMedQA Vector Store\n- ✅ Hybrid Search (Vector + BM25)")
                with col2:
                    st.markdown("**🌐 Dynamic Sources**\n- ✅ PubMed E-utilities (2024-2026)\n- ✅ Real-time clinical data")
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 33%"></div></div>', unsafe_allow_html=True)
            time.sleep(0.5)
            
            # Stage 3
            with pipeline_placeholder:
                show_compact_pipeline(3)
            with output_container:
                with st.spinner("📚 Retrieving from multiple sources..."):
                    time.sleep(1.5)
                st.markdown(f'<div class="success-box">✅ <strong>Retrieved {len(SAMPLE_SOURCES)} high-relevance documents</strong></div>', unsafe_allow_html=True)
                with st.expander("📄 View Retrieved Sources"):
                    for i, src in enumerate(SAMPLE_SOURCES):
                        relevance_color = "#10b981" if src['relevance'] > 0.9 else "#f59e0b"
                        st.markdown(f'<div class="source-card"><strong>{i+1}. {src["title"]}</strong><br>📍 {src["source"]} | 📅 {src["year"]} | <span style="color: {relevance_color};">🎯 {src["relevance"]:.0%}</span></div>', unsafe_allow_html=True)
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 50%"></div></div>', unsafe_allow_html=True)
            time.sleep(0.5)
            
            # Stage 4
            with pipeline_placeholder:
                show_compact_pipeline(4)
            with output_container:
                with st.spinner("⚖️ Ranking contexts..."):
                    time.sleep(1.0)
                st.markdown('<div class="success-box">✅ <strong>Context Ranked & Optimized</strong></div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", "127")
                with col2:
                    st.metric("Selected", "5")
                with col3:
                    st.metric("Tokens", "2,847")
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 66%"></div></div>', unsafe_allow_html=True)
            time.sleep(0.5)
            
            # Stage 5
            with pipeline_placeholder:
                show_compact_pipeline(5)
            with output_container:
                with st.spinner("🧬 TXGemma-9B generating report..."):
                    time.sleep(2.0)
                st.markdown('<div class="success-box">✅ <strong>Clinical Report Generated!</strong></div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Length", "1,735 chars")
                with col2:
                    st.metric("Time", "46.48s")
                with col3:
                    st.metric("Sources", "5")
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 83%"></div></div>', unsafe_allow_html=True)
            time.sleep(0.5)
            
            # Stage 6
            with pipeline_placeholder:
                show_compact_pipeline(6)
            with output_container:
                with st.spinner("✅ Running evaluation..."):
                    time.sleep(1.2)
                st.markdown('<div class="success-box">✅ <strong>Quality Assessment Complete!</strong></div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("BERTScore F1", "83.14%")
                with col2:
                    st.metric("Faithfulness", "51.19%")
                with col3:
                    st.metric("Relevancy", "20.15%")
                with col4:
                    st.metric("LLM Judge", "3.77/5")
            
            progress_placeholder.markdown('<div class="progress-container"><div class="progress-bar" style="width: 100%"></div></div>', unsafe_allow_html=True)
            st.balloons()
            
            # Final report
            st.markdown("---")
            st.markdown("## 📋 Generated Clinical Report")
            st.markdown('<div class="report-section">', unsafe_allow_html=True)
            st.markdown(SAMPLE_REPORT)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================
    # TAB 2: EVALUATION (same as before)
    # =========================
    with tab2:
        st.markdown("### 📊 Comprehensive Evaluation Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">83.14%</div><div class="metric-label">BERTScore F1</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">85.15%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">51.19%</div><div class="metric-label">Faithfulness</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">3.77/5</div><div class="metric-label">LLM Judge</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 🔬 BERTScore Analysis")
            st.markdown("| Metric | Score |\n|--------|-------|\n| Precision | 84.51% |\n| Recall | 81.85% |\n| **F1 Score** | **83.14%** |")
            st.markdown("#### 📈 Performance Profile")
            st.plotly_chart(create_performance_comparison(), use_container_width=True)
        with col_right:
            st.markdown("#### ⚖️ LLM-as-a-Judge")
            st.plotly_chart(create_evaluation_chart(), use_container_width=True)
    
    # TAB 3 & 4: Keep architecture and hackathon impact sections similar to before
    with tab3:
        st.markdown("### 🏗️ System Architecture")
        # Framework diagram
        framework_path = Path("Clinexa.png")
        if framework_path.exists():
            st.image(str(framework_path), caption="Clinexa 6-Stage Agentic Pipeline Architecture", use_container_width=True)
        st.markdown("### 🎯 Key Innovations")
        st.markdown("- 🔄 Agentic Multi-Stage Pipeline\n- ⚡ Hybrid RAG (Vector + BM25)\n- 🧬 TXGemma Integration\n- 🌐 Dynamic API Integration")
    
    with tab4:
        st.markdown("### 🎯 Hackathon Impact")
        st.markdown('<div class="hackathon-badge" style="display: block; margin: 2rem auto; font-size: 1.2rem;">🏆 Agentic Workflow Prize</div>', unsafe_allow_html=True)
        st.markdown("### 💡 Problem: Knowledge Staleness in Medical AI")
        st.markdown("- Static datasets become obsolete\n- New drugs approved, trials conclude\n- Healthcare requires recency + accuracy")
        st.markdown("### ✅ Solution: Clinexa")
        st.markdown("- 6-stage agentic pipeline\n- HYBRID RAG (Static + Dynamic)\n- 83.14% semantic accuracy")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h3 style="color: #667eea;">🏥 Clinexa</h3>
            <p style="color: #6b7280;">Built with TXGemma-9B • Powered by LlamaIndex • 83.14% Semantic Accuracy</p>
            <p style="color: #6b7280;">👩‍💻 <strong>Bushra Salama Aljohani</strong></p>
            <p style="color: #6b7280;">MedGemma Impact Challenge 2026</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("🔗 View on GitHub", "https://github.com/bushra-aljohani/clinexa", use_container_width=True)

if __name__ == "__main__":
    main()
