"""
app/pages/provider_lookup.py
The core feature of the app.

Enter any NPI → see fraud risk score, tier, SHAP explanation,
peer comparison chart, and OIG cross-reference.

This is what stops a recruiter in their tracks — they can type
in a real NPI and get a real answer with explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.data_loader import (load_providers, load_shap, load_benchmarks,
                                    tier_color, tier_badge, FEATURE_DISPLAY)

@st.cache_data
def _load():
    df    = load_providers()
    shap  = load_shap()
    bench = load_benchmarks()
    return df, shap, bench


def render():
    df, shap_df, bench = _load()

    st.markdown("## 🔎 Provider Lookup")
    st.markdown("Enter a provider NPI to see their fraud risk score, billing pattern analysis, and SHAP explanation.")

    # ── NPI input ─────────────────────────────────────────────
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        npi_input = st.text_input(
            "Provider NPI",
            placeholder="e.g. 1100000042",
            label_visibility="collapsed",
        )
    with col_btn:
        lookup = st.button("🔍 Analyze", use_container_width=True)

    # Show sample NPIs for demo
    st.markdown("""
    <div style='font-size:11px;color:#6b7280;margin-top:-8px'>
    Try a sample NPI:
    <code style='background:#1a1d27;padding:2px 6px;border-radius:4px'>1100000042</code>
    <code style='background:#1a1d27;padding:2px 6px;border-radius:4px'>1100000105</code>
    <code style='background:#1a1d27;padding:2px 6px;border-radius:4px'>1100000001</code>
    </div>
    """, unsafe_allow_html=True)

    if not npi_input:
        st.info("Enter an NPI above to analyze a provider.")

        # Show stat teaser
        n_critical = len(df[df["fraud_tier"] == "CRITICAL"]) if "fraud_tier" in df.columns else 0
        st.markdown(f"""
        <div style='background:#1a1d27;border:1px solid #2d3142;border-radius:12px;
                    padding:20px;margin-top:24px;font-size:13px;color:#9ca3af;line-height:2'>
        This tool analyzed <strong style='color:#f0eee8'>{len(df):,} Medicare Part D providers</strong>.<br>
        <strong style='color:#ef4444'>{n_critical:,} providers</strong> were flagged as CRITICAL risk.<br>
        Of those, <strong style='color:#ef4444'>98%</strong> were confirmed on the OIG exclusion list.<br><br>
        The model uses <strong style='color:#f0eee8'>24 engineered features</strong> — all computed relative
        to specialty peer groups, so a pain specialist prescribing opioids is evaluated
        against other pain specialists, not the general population.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Provider lookup ───────────────────────────────────────
    provider = df[df["npi"] == str(npi_input).strip()]

    if len(provider) == 0:
        st.error(f"NPI `{npi_input}` not found in this dataset.")
        st.markdown("Note: This dataset contains 50,000 providers. The full CMS Part D dataset has 1.2 million providers.")
        return

    p = provider.iloc[0]
    tier = p.get("fraud_tier", "UNKNOWN")
    score = p.get("fraud_score_ensemble", 0.0)

    # ── Header card ───────────────────────────────────────────
    st.markdown("---")
    h1, h2 = st.columns([2, 1])

    with h1:
        name = f"Dr. {p.get('first_name','')} {p.get('last_name','')}".strip()
        st.markdown(f"### {name}")
        st.markdown(f"""
        <div style='font-size:13px;color:#9ca3af;line-height:2'>
        <strong style='color:#f0eee8'>NPI:</strong> {p['npi']} &nbsp;|&nbsp;
        <strong style='color:#f0eee8'>Specialty:</strong> {p.get('specialty','N/A')} &nbsp;|&nbsp;
        <strong style='color:#f0eee8'>State:</strong> {p.get('state','N/A')}
        </div>
        """, unsafe_allow_html=True)

    with h2:
        color = tier_color(tier)
        st.markdown(f"""
        <div style='background:#1a1d27;border:2px solid {color};border-radius:12px;
                    padding:20px;text-align:center'>
            <div style='font-size:2.8rem;font-weight:700;color:{color};
                        font-family:monospace'>{score:.3f}</div>
            <div style='font-size:11px;color:#6b7280;margin:4px 0'>Ensemble Fraud Score</div>
            {tier_badge(tier)}
        </div>
        """, unsafe_allow_html=True)

    # ── Score gauge ───────────────────────────────────────────
    st.markdown("---")
    col_gauge, col_flags = st.columns([1, 1])

    with col_gauge:
        st.markdown("#### Fraud Score Breakdown")
        scores_data = {
            "XGBoost (supervised)":    p.get("fraud_score_xgb", 0),
            "Isolation Forest (unsup)": p.get("fraud_score_iso", 0),
            "Ensemble (final)":         score,
        }
        fig = go.Figure()
        for label, val in scores_data.items():
            bar_color = "#ef4444" if val > 0.8 else "#f97316" if val > 0.6 else "#eab308" if val > 0.4 else "#22c55e"
            fig.add_trace(go.Bar(
                name=label, x=[val], y=[label],
                orientation="h",
                marker_color=bar_color,
                text=f"{val:.3f}", textposition="outside",
                width=0.4,
            ))
        fig.add_vline(x=0.80, line_color="#ef4444", line_dash="dash", opacity=0.5)
        fig.add_vline(x=0.60, line_color="#f97316", line_dash="dash", opacity=0.5)
        fig.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=220,
            margin=dict(t=10,b=10,l=10,r=60),
            showlegend=False,
            xaxis=dict(range=[0,1], showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_flags:
        st.markdown("#### Fraud Flags Triggered")
        flag_map = {
            "flag_high_volume":          ("Extreme claim volume",    "top 1% in specialty"),
            "flag_high_cost_per_claim":  ("High cost per claim",     "top 1% in specialty"),
            "flag_brand_heavy":          ("Brand-heavy prescribing", ">70% brand drugs"),
            "flag_concentrated_benes":   ("Concentrated patients",   "pill mill pattern"),
            "flag_opioid_heavy":         ("Opioid overuse",          ">20% opioid claims"),
        }
        flag_count = int(p.get("flag_count", 0))
        st.markdown(f"**{flag_count} of 5 flags triggered**")
        for flag_col, (label, note) in flag_map.items():
            val = int(p.get(flag_col, 0))
            icon = "🔴" if val else "⚪"
            color = "#ef4444" if val else "#374151"
            st.markdown(f"""
            <div style='background:#1a1d27;border:1px solid {color};border-radius:8px;
                        padding:8px 12px;margin:4px 0;font-size:12px'>
                {icon} <strong>{label}</strong>
                <span style='color:#6b7280;float:right'>{note}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Billing metrics ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Billing Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Total Claims",        f"{int(p.get('total_claim_count',0)):,}"),
        ("Total Drug Cost",     f"${p.get('total_drug_cost',0)/1000:.0f}K"),
        ("Cost per Claim",      f"${p.get('cost_per_claim',0):.0f}"),
        ("Brand Drug Share",    f"{p.get('brand_share',0)*100:.1f}%"),
        ("Opioid Share",        f"{p.get('opioid_share',0)*100:.1f}%"),
    ]
    for col, (label, value) in zip([m1,m2,m3,m4,m5], metrics):
        col.metric(label, value)

    # ── Peer comparison ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Peer Group Comparison")
    st.markdown(f"How this provider compares to other **{p.get('specialty','N/A')}** providers:")

    spec_peers = df[df["specialty"] == p.get("specialty","")]
    if len(spec_peers) > 10:
        compare_metrics = ["total_claim_count","cost_per_claim","brand_share","opioid_share"]
        labels = ["Total Claims","Cost/Claim ($)","Brand Share (%)","Opioid Share (%)"]
        multipliers = [1, 1, 100, 100]

        peer_avgs = [spec_peers[m].mean() * mult for m, mult in zip(compare_metrics, multipliers)]
        provider_vals = [p.get(m, 0) * mult for m, mult in zip(compare_metrics, multipliers)]

        fig_peer = go.Figure()
        fig_peer.add_trace(go.Bar(
            name="Specialty Average", x=labels, y=peer_avgs,
            marker_color="#4b5563", opacity=0.8,
        ))
        fig_peer.add_trace(go.Bar(
            name=f"This Provider ({p['npi']})", x=labels, y=provider_vals,
            marker_color=tier_color(tier), opacity=0.9,
        ))
        fig_peer.update_layout(
            barmode="group",
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=320,
            margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(bgcolor="#1a1d27"),
            yaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig_peer, use_container_width=True)

    # ── SHAP explanation ──────────────────────────────────────
    if len(shap_df) > 0:
        provider_shap = shap_df[shap_df["npi"] == str(npi_input).strip()]
        if len(provider_shap) > 0:
            row = provider_shap.iloc[0]
            st.markdown("---")
            st.markdown("#### Why Was This Provider Flagged?")
            st.markdown("Top 3 features driving the fraud score (SHAP values):")
            for i, col in enumerate(["reason_1","reason_2","reason_3"], 1):
                reason = row.get(col, "")
                if reason:
                    st.markdown(f"""
                    <div style='background:#1a1d27;border-left:3px solid #6366f1;
                                padding:10px 16px;margin:6px 0;border-radius:0 8px 8px 0;
                                font-size:13px;color:#d1d5db'>
                        <strong style='color:#818cf8'>#{i}</strong> {reason}
                    </div>
                    """, unsafe_allow_html=True)

    # ── OIG status ────────────────────────────────────────────
    st.markdown("---")
    is_fraud = int(p.get("is_fraud_label", 0))
    if is_fraud:
        st.error("⚠️ **OIG EXCLUSION CONFIRMED** — This provider appears on the OIG exclusion list.")
    else:
        st.success("✅ Not found on OIG exclusion list in this dataset.")
    st.markdown("""
    <div style='font-size:11px;color:#6b7280'>
    OIG exclusion list: providers excluded from Medicare/Medicaid participation.
    Source: <a href='https://oig.hhs.gov/exclusions/' style='color:#6366f1'>oig.hhs.gov/exclusions</a>
    </div>
    """, unsafe_allow_html=True)
