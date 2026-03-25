"""
app/pages/state_analysis.py
Geographic fraud analysis — choropleth map + state breakdown table.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.data_loader import load_providers


@st.cache_data
def _load():
    return load_providers()


def render():
    df = _load()

    st.markdown("## 🗺️ State-Level Fraud Analysis")
    st.markdown("Geographic distribution of Medicare fraud risk across the United States.")

    if "fraud_tier" not in df.columns:
        st.error("Fraud scores not available.")
        return

    # ── State aggregation ─────────────────────────────────────
    state_stats = df.groupby("state").agg(
        total_providers   = ("npi",                  "count"),
        critical_count    = ("fraud_tier",           lambda x: (x == "CRITICAL").sum()),
        high_count        = ("fraud_tier",            lambda x: (x == "HIGH").sum()),
        avg_fraud_score   = ("fraud_score_ensemble",  "mean"),
        total_drug_cost_M = ("total_drug_cost",       lambda x: x.sum() / 1e6),
        oig_confirmed     = ("is_fraud_label",        "sum"),
    ).reset_index()

    state_stats["high_risk_pct"] = (
        (state_stats["critical_count"] + state_stats["high_count"])
        / state_stats["total_providers"] * 100
    ).round(2)

    state_stats["high_risk_total"] = (
        state_stats["critical_count"] + state_stats["high_count"]
    )

    # ── Map metric selector ───────────────────────────────────
    map_metric = st.selectbox(
        "Map metric",
        ["High-Risk Provider %", "Average Fraud Score", "CRITICAL Count", "Total Drug Cost ($M)"],
    )

    metric_map = {
        "High-Risk Provider %":    ("high_risk_pct",       "% of Providers High-Risk"),
        "Average Fraud Score":     ("avg_fraud_score",     "Avg Fraud Score"),
        "CRITICAL Count":          ("critical_count",       "CRITICAL Providers"),
        "Total Drug Cost ($M)":    ("total_drug_cost_M",   "Drug Cost ($M)"),
    }
    col, label = metric_map[map_metric]

    # ── Choropleth map ────────────────────────────────────────
    fig = px.choropleth(
        state_stats,
        locations="state",
        locationmode="USA-states",
        color=col,
        color_continuous_scale=["#1a1d27", "#f97316", "#ef4444"],
        scope="usa",
        labels={col: label},
        hover_data={
            "state": True,
            "total_providers": True,
            "critical_count": True,
            "high_risk_pct": ":.1f",
            "avg_fraud_score": ":.3f",
            "oig_confirmed": True,
        },
    )
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#f0eee8",
        geo=dict(bgcolor="#0f1117", lakecolor="#0f1117",
                 landcolor="#1a1d27", showlakes=True),
        height=450,
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(
            bgcolor="#1a1d27",
            tickfont=dict(color="#f0eee8"),
            title=dict(text=label, font=dict(color="#f0eee8")),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── State ranking table ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### State Rankings")

    sort_by = st.selectbox(
        "Sort by",
        ["High-Risk %", "CRITICAL Count", "Avg Fraud Score", "Total Drug Cost"],
    )
    sort_col_map = {
        "High-Risk %":      "high_risk_pct",
        "CRITICAL Count":   "critical_count",
        "Avg Fraud Score":  "avg_fraud_score",
        "Total Drug Cost":  "total_drug_cost_M",
    }
    sorted_states = state_stats.sort_values(
        sort_col_map[sort_by], ascending=False
    ).reset_index(drop=True)
    sorted_states.index += 1

    # Format for display
    display = sorted_states.copy()
    display["high_risk_pct"]     = display["high_risk_pct"].apply(lambda x: f"{x:.1f}%")
    display["avg_fraud_score"]   = display["avg_fraud_score"].apply(lambda x: f"{x:.4f}")
    display["total_drug_cost_M"] = display["total_drug_cost_M"].apply(lambda x: f"${x:.1f}M")
    display = display.rename(columns={
        "state": "State", "total_providers": "Providers",
        "critical_count": "CRITICAL", "high_count": "HIGH",
        "high_risk_pct": "High-Risk %", "avg_fraud_score": "Avg Score",
        "total_drug_cost_M": "Drug Cost",
        "oig_confirmed": "OIG Confirmed",
    })
    st.dataframe(display[[
        "State","Providers","CRITICAL","HIGH","High-Risk %",
        "Avg Score","Drug Cost","OIG Confirmed"
    ]], use_container_width=True, height=400)

    # ── Two bottom charts ─────────────────────────────────────
    st.markdown("---")
    bc1, bc2 = st.columns(2)

    with bc1:
        st.markdown("#### Top 10 States by CRITICAL Providers")
        top10 = state_stats.nlargest(10, "critical_count")
        fig2 = go.Figure(go.Bar(
            x=top10["state"], y=top10["critical_count"],
            marker_color="#ef4444", opacity=0.88,
            text=top10["critical_count"], textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=300,
            margin=dict(t=10,b=10,l=10,r=10),
            yaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with bc2:
        st.markdown("#### High-Risk % vs Total Providers")
        fig3 = go.Figure(go.Scatter(
            x=state_stats["total_providers"],
            y=state_stats["high_risk_pct"],
            mode="markers+text",
            text=state_stats["state"],
            textposition="top center",
            textfont=dict(size=9, color="#9ca3af"),
            marker=dict(
                size=state_stats["critical_count"].clip(lower=2) * 0.8,
                color=state_stats["avg_fraud_score"],
                colorscale=["#22c55e","#ef4444"],
                showscale=True,
                colorbar=dict(title="Avg Score", thickness=10),
            ),
        ))
        fig3.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=300,
            margin=dict(t=10,b=10,l=10,r=10),
            xaxis_title="Total Providers",
            yaxis_title="High-Risk %",
            xaxis=dict(showgrid=True, gridcolor="#2d3142"),
            yaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig3, use_container_width=True)
