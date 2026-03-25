"""
app/pages/drug_analysis.py
Drug category analysis — which drug patterns correlate with fraud.
Brand vs generic, opioid concentration, cost outliers by specialty.
"""

import streamlit as st
import pandas as pd
import numpy as np
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

    st.markdown("## 💊 Drug Prescribing Pattern Analysis")
    st.markdown(
        "Brand vs generic ratios, opioid concentration, and cost-per-claim outliers "
        "by specialty. Fraudulent providers consistently show brand-heavy, opioid-heavy, "
        "and high-cost-per-claim patterns."
    )

    if "fraud_tier" not in df.columns:
        st.error("Fraud scores not available.")
        return

    # ── Brand vs generic analysis ─────────────────────────────
    st.markdown("---")
    st.markdown("#### Brand Drug Share — Fraud vs Normal Providers")
    st.markdown(
        "The national average brand drug share is ~28%. "
        "Fraudulent providers average **82%** brand. "
        "This is the clearest single signal in the dataset."
    )

    bc1, bc2 = st.columns(2)

    with bc1:
        # Distribution by tier
        tiers_plot = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        colors_plot = ["#22c55e", "#eab308", "#f97316", "#ef4444"]

        fig = go.Figure()
        for tier, color in zip(tiers_plot, colors_plot):
            subset = df[df["fraud_tier"] == tier]["brand_share"].dropna() * 100
            if len(subset) > 0:
                fig.add_trace(go.Box(
                    y=subset, name=tier,
                    marker_color=color,
                    line_color=color,
                    boxmean=True,
                ))
        fig.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=360,
            margin=dict(t=10,b=10,l=10,r=10),
            yaxis_title="Brand Drug Share (%)",
            yaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        fig.add_hline(y=28, line_dash="dash", line_color="#6b7280",
                      annotation_text="National avg 28%", annotation_font_color="#6b7280")
        st.plotly_chart(fig, use_container_width=True)

    with bc2:
        # Mean brand share per tier
        brand_by_tier = df.groupby("fraud_tier")["brand_share"].agg(["mean","median"]).reset_index()
        brand_by_tier["mean_pct"]   = brand_by_tier["mean"]   * 100
        brand_by_tier["median_pct"] = brand_by_tier["median"] * 100
        brand_by_tier = brand_by_tier.set_index("fraud_tier").reindex(
            ["LOW","MODERATE","HIGH","CRITICAL"]
        ).reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Mean", x=brand_by_tier["fraud_tier"],
            y=brand_by_tier["mean_pct"],
            marker_color=["#22c55e","#eab308","#f97316","#ef4444"],
            text=[f"{v:.1f}%" for v in brand_by_tier["mean_pct"]],
            textposition="outside",
        ))
        fig2.add_hline(y=28, line_dash="dash", line_color="#6b7280",
                       annotation_text="National avg 28%")
        fig2.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=360,
            margin=dict(t=10,b=10,l=10,r=10),
            yaxis_title="Mean Brand Share (%)",
            yaxis=dict(showgrid=True, gridcolor="#2d3142"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Opioid analysis ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Opioid Prescribing Share by Specialty")
    st.markdown("High opioid share is a key fraud signal, especially for non-pain specialties.")

    oc1, oc2 = st.columns(2)
    with oc1:
        spec_opioid = df.groupby("specialty")["opioid_share"].mean().sort_values(ascending=False) * 100
        fig3 = px.bar(
            x=spec_opioid.values, y=spec_opioid.index,
            orientation="h",
            color=spec_opioid.values,
            color_continuous_scale=["#22c55e", "#f97316", "#ef4444"],
        )
        fig3.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=420,
            margin=dict(t=10,b=10,l=10,r=10),
            coloraxis_showscale=False,
            xaxis_title="Mean Opioid Share (%)",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with oc2:
        # Opioid share vs fraud score scatter
        sample = df.sample(min(2000, len(df)), random_state=42)
        colors = [tier_color(t) for t in sample["fraud_tier"].fillna("LOW")]
        fig4 = go.Figure(go.Scatter(
            x=sample["opioid_share"] * 100,
            y=sample["fraud_score_ensemble"],
            mode="markers",
            marker=dict(color=colors, size=4, opacity=0.5),
            text=sample["specialty"],
            hovertemplate="<b>%{text}</b><br>Opioid%: %{x:.1f}<br>Score: %{y:.3f}",
        ))
        fig4.add_vline(x=20, line_dash="dash", line_color="#ef4444",
                       annotation_text="Flag threshold 20%", annotation_font_color="#ef4444")
        fig4.update_layout(
            paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
            font_color="#f0eee8", height=420,
            margin=dict(t=10,b=10,l=10,r=10),
            xaxis_title="Opioid Share (%)",
            yaxis_title="Fraud Score",
            xaxis=dict(showgrid=True, gridcolor="#2d3142"),
            yaxis=dict(showgrid=True, gridcolor="#2d3142"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Cost per claim analysis ───────────────────────────────
    st.markdown("---")
    st.markdown("#### Cost per Claim — Extreme Outliers")
    st.markdown("Fraudulent providers charge significantly more per claim vs specialty peers.")

    cost_by_tier = df.groupby("fraud_tier").agg(
        avg_cost_per_claim = ("cost_per_claim", "mean"),
        p95_cost_per_claim = ("cost_per_claim", lambda x: x.quantile(0.95)),
    ).reindex(["LOW","MODERATE","HIGH","CRITICAL"]).reset_index()

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        name="Mean", x=cost_by_tier["fraud_tier"],
        y=cost_by_tier["avg_cost_per_claim"],
        marker_color=["#22c55e","#eab308","#f97316","#ef4444"],
        opacity=0.9,
        text=[f"${v:.0f}" for v in cost_by_tier["avg_cost_per_claim"]],
        textposition="outside",
    ))
    fig5.update_layout(
        paper_bgcolor="#1a1d27", plot_bgcolor="#1a1d27",
        font_color="#f0eee8", height=320,
        margin=dict(t=10,b=40,l=10,r=10),
        yaxis_title="Mean Cost per Claim ($)",
        xaxis_title="Fraud Tier",
        yaxis=dict(showgrid=True, gridcolor="#2d3142"),
        showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ── Key takeaways ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Key Pattern Summary")
    fraud_prov = df[df["fraud_tier"] == "CRITICAL"]
    normal_prov = df[df["fraud_tier"] == "LOW"]

    if len(fraud_prov) > 0 and len(normal_prov) > 0:
        rows = [
            ("Brand drug share",  f"{normal_prov['brand_share'].mean()*100:.1f}%",
             f"{fraud_prov['brand_share'].mean()*100:.1f}%",
             f"{fraud_prov['brand_share'].mean()/normal_prov['brand_share'].mean():.1f}x"),
            ("Opioid share",      f"{normal_prov['opioid_share'].mean()*100:.1f}%",
             f"{fraud_prov['opioid_share'].mean()*100:.1f}%",
             f"{fraud_prov['opioid_share'].mean()/max(normal_prov['opioid_share'].mean(),0.001):.1f}x"),
            ("Cost per claim",    f"${normal_prov['cost_per_claim'].mean():.0f}",
             f"${fraud_prov['cost_per_claim'].mean():.0f}",
             f"{fraud_prov['cost_per_claim'].mean()/max(normal_prov['cost_per_claim'].mean(),1):.1f}x"),
            ("Claims per patient",f"{normal_prov['claims_per_bene'].mean():.1f}",
             f"{fraud_prov['claims_per_bene'].mean():.1f}",
             f"{fraud_prov['claims_per_bene'].mean()/max(normal_prov['claims_per_bene'].mean(),0.001):.1f}x"),
        ]
        summary = pd.DataFrame(rows, columns=["Metric","Normal (LOW tier)","Fraud (CRITICAL tier)","Ratio"])
        st.dataframe(summary, use_container_width=True, hide_index=True)
