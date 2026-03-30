"""
streamlit_app.py — Streamlit Cloud entry point
"""
import sys
import os
from pathlib import Path

# Add repo root to path so all imports work
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Set up working directory so relative file paths work
os.chdir(ROOT)

import streamlit as st

st.set_page_config(
    page_title="Medicare Fraud Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background: #0f1117; }
    .stApp { background: #0f1117; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #f0eee8; font-family: monospace; }
    .metric-label { font-size: 12px; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    section[data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #2d3142; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔍 Medicare Fraud Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Provider Lookup", "High-Risk Dashboard", "State Analysis", "Drug Analysis"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#6b7280;line-height:1.8'>
    <strong style='color:#9ca3af'>Data</strong><br>
    CMS Medicare Part D 2022<br>
    OIG Exclusions List<br><br>
    <strong style='color:#9ca3af'>Model</strong><br>
    XGBoost + Isolation Forest<br>
    PR-AUC: 0.9996<br>
    OIG Confirmation: 98%<br><br>
    <strong style='color:#9ca3af'>Built by</strong><br>
    Divya Dhole · MS Data Science<br>
    <a href='https://divyadhole.github.io' style='color:#6366f1'>Portfolio</a> ·
    <a href='https://www.linkedin.com/in/divyadhole/' style='color:#6366f1'>LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

if page == "Overview":
    from app.pages.overview import render
    render()
elif page == "Provider Lookup":
    from app.pages.provider_lookup import render
    render()
elif page == "High-Risk Dashboard":
    from app.pages.high_risk import render
    render()
elif page == "State Analysis":
    from app.pages.state_analysis import render
    render()
elif page == "Drug Analysis":
    from app.pages.drug_analysis import render
    render()
