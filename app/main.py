"""
app/main.py
Medicare Fraud Intelligence Dashboard
Entry point for Streamlit app.

Run locally:
    streamlit run app/main.py

Deployed at:
    https://cms-medicare-fraud-intelligence.streamlit.app
"""

import streamlit as st

st.set_page_config(
    page_title="Medicare Fraud Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark clean theme */
    .main { background: #0f1117; }
    .stApp { background: #0f1117; }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f0eee8;
        font-family: 'DM Mono', monospace;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Risk tier badges */
    .tier-critical { background:#ef4444; color:#fff; padding:4px 14px;
                     border-radius:99px; font-weight:700; font-size:13px; }
    .tier-high     { background:#f97316; color:#fff; padding:4px 14px;
                     border-radius:99px; font-weight:700; font-size:13px; }
    .tier-moderate { background:#eab308; color:#000; padding:4px 14px;
                     border-radius:99px; font-weight:700; font-size:13px; }
    .tier-low      { background:#22c55e; color:#fff; padding:4px 14px;
                     border-radius:99px; font-weight:700; font-size:13px; }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        color: #6b7280;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
        padding-top: 16px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #13151f;
        border-right: 1px solid #2d3142;
    }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
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
    <strong style='color:#9ca3af'>Data Source</strong><br>
    CMS Medicare Part D 2022<br>
    FEMA OpenFEMA API<br>
    OIG Exclusions List<br><br>
    <strong style='color:#9ca3af'>Model</strong><br>
    XGBoost + Isolation Forest<br>
    PR-AUC: 0.9996<br>
    OIG Confirmation: 98%<br><br>
    <strong style='color:#9ca3af'>Built by</strong><br>
    Divya Dhole<br>
    MS Data Science @ UArizona<br>
    <a href='https://divyadhole.github.io' style='color:#6366f1'>Portfolio</a> ·
    <a href='https://www.linkedin.com/in/divyadhole/' style='color:#6366f1'>LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ── Route pages ───────────────────────────────────────────────
if page == "Overview":
    from app.pages import overview
    overview.render()
elif page == "Provider Lookup":
    from app.pages import provider_lookup
    provider_lookup.render()
elif page == "High-Risk Dashboard":
    from app.pages import high_risk
    high_risk.render()
elif page == "State Analysis":
    from app.pages import state_analysis
    state_analysis.render()
elif page == "Drug Analysis":
    from app.pages import drug_analysis
    drug_analysis.render()
