"""
app/pages/high_risk.py
High-risk provider dashboard.
Top 100 CRITICAL + HIGH providers, filterable by state and specialty.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.data_loader import load_providers, tier_color


@st.cache_data
def _load():
    return load_providers()


def render():
    df = _load()

    st.markdown("## ⚠️ High-Risk Provider Dashboard")
    st.markdown("Top flagged providers across CRITICAL and HIGH risk tiers. Filter by state or specialty.")

    if "fraud_tier" not in df.columns:
        st.error("Fraud scores not available. Run model pipeline first.")
        return

    # ── Filters ───────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 1])

    with fc1:
        tier_filter = st.multiselect(
            "Risk Tier",
            ["CRITICAL", "HIGH", "MODERATE"],
            default=["CRITICAL", "HIGH"],
        )
    with fc2:
        states = sorted(df["state"].dropna().unique().tolist())
        state_filter = st.multiselect("State", states, default=[])
    with fc3:
        specs = sorted(df["specialty"].dropna().unique().tolist())
        spec_filter = st.multiselect("Specialty", specs, default=[])

    # ── Filter data ───────────────────────────────────────────
    filtered = df[df["fraud_tier"].isin(tier_filter)].copy()
    if state_filter:
        filtered = filtered[filtered["state"].isin(state_filter)]
    if spec_filter:
        filtered = filtered[filtered["specialty"].isin(spec_filter)]

    filtered = filtered.sort_values("fraud_score_ensemble", ascending=False).head(100)

    st.markdown(f"**{len(filtered):,} providers** matching filters")

    # ── Summary cards ─────────────────────────────────────────
    if len(filtered) > 0:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            total_cost = filtered["total_drug_cost"].sum()
            st.metric("Total At-Risk Spend", f"${total_cost/1e6:.1f}M")
        with s2:
            avg_score = filtered["fraud_score_ensemble"].mean()
            st.metric("Avg Fraud Score", f"{avg_score:.3f}")
        with s3:
            oig_count = int(filtered["is_fraud_label"].sum()) \
                        if "is_fraud_label" in filtered.columns else "N/A"
            st.metric("OIG Confirmed", str(oig_count))
        with s4:
            avg_flags = filtered["flag_count"].mean() \
                        if "flag_count" in filtered.columns else 0
            st.metric("Avg Flags", f"{avg_flags:.1f} / 5")

    st.markdown("---")

    # ── Table ─────────────────────────────────────────────────
    display_cols = ["npi", "first_name", "last_name", "specialty", "state",
                    "fraud_tier", "fraud_score_ensemble",
                    "total_claim_count", "total_drug_cost",
                    "cost_per_claim", "brand_share", "opioid_share",
                    "flag_count", "is_fraud_label"]
    available = [c for c in display_cols if c in filtered.columns]
    table = filtered[available].copy()

    # Format for display
    if "total_drug_cost" in table.columns:
        table["total_drug_cost"] = table["total_drug_cost"].apply(lambda x: f"${x/1000:.0f}K")
    if "cost_per_claim" in table.columns:
        table["cost_per_claim"] = table["cost_per_claim"].apply(lambda x: f"${x:.0f}")
    if "brand_share" in table.columns:
        table["brand_share"] = table["brand_share"].apply(lambda x: f"{x*100:.1f}%")
    if "opioid_share" in table.columns:
        table["opioid_share"] = table["opioid_share"].apply(lambda x: f"{x*100:.1f}%")
    if "fraud_score_ensemble" in table.columns:
        table["fraud_score_ensemble"] = table["fraud_score_ensemble"].apply(lambda x: f"{x:.4f}")
    if "is_fraud_label" in table.columns:
        table["is_fraud_label"] = table["is_fraud_label"].apply(
            lambda x: "✅ OIG Confirmed" if x == 1 else "")

    table = table.rename(columns={
        "npi": "NPI", "first_name": "First", "last_name": "Last",
        "specialty": "Specialty", "state": "State",
        "fraud_tier": "Tier", "fraud_score_ensemble": "Score",
        "total_claim_count": "Claims", "total_drug_cost": "Drug Cost",
        "cost_per_claim": "Cost/Claim", "brand_share": "Brand%",
        "opioid_share": "Opioid%", "flag_count": "Flags",
        "is_fraud_label": "OIG Status",
    })

    st.dataframe(table, use_container_width=True, height=480)

    # ── Charts ────────────────────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### Flagged Providers by State")
        orig_filtered = df[df["fraud_tier"].isin(tier_filter)]
        if state_filter:
            orig_filtered = orig_filtered[orig_filtered["state"].isin(state_filter)]
        state_counts = orig_filtered["state"].value_counts().head(15).reset_index()
        state_counts.columns = ["State","Count"]
        fig = px.bar(state_counts, x="Count", y="State", orientation="h",
                     color="Count", color_continuous_scale=["#f97316","#ef4444"])
        fig.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=380, coloraxis_showscale=False,
            margin=dict(t=10,b=10,l=10,r=10),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("#### Fraud Score vs Drug Cost")
        plot_df = df[df["fraud_tier"].isin(tier_filter)].head(500)
        if len(plot_df) > 0:
            colors = [tier_color(t) for t in plot_df["fraud_tier"]]
            fig2 = go.Figure(go.Scatter(
                x=plot_df["total_drug_cost"]/1000,
                y=plot_df["fraud_score_ensemble"],
                mode="markers",
                marker=dict(color=colors, size=6, opacity=0.7),
                text=plot_df["specialty"],
                hovertemplate="<b>%{text}</b><br>Score: %{y:.3f}<br>Cost: $%{x:.0f}K",
            ))
            fig2.update_layout(
                paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
                font_color="#f0eee8", height=380,
                margin=dict(t=10,b=10,l=10,r=10),
                xaxis_title="Total Drug Cost ($K)",
                yaxis_title="Fraud Score",
                yaxis=dict(showgrid=True, gridcolor="#2d3142"),
                xaxis=dict(showgrid=True, gridcolor="#2d3142"),
            )
            st.plotly_chart(fig2, use_container_width=True)
