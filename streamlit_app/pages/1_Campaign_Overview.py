"""
Campaign Overview Page
Treatment effect analysis and overall campaign performance.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, COLORS
from utils.data_loader import (
    load_eda_summary,
    load_outcome_distribution,
    load_uplift_by_quartile,
    load_image
)

st.set_page_config(**PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1565C0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1A1A2E;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("Campaign Overview")
    st.markdown("*Treatment effect analysis and overall campaign performance*")
    
    # Load data
    eda_summary = load_eda_summary()
    outcome_dist = load_outcome_distribution()
    uplift_quartile = load_uplift_by_quartile()
    
    # Extract key metrics from EDA summary
    def get_metric(metric_name):
        row = eda_summary[eda_summary['Metric'] == metric_name]
        return row['Value'].values[0] if len(row) > 0 else None
    
    control_pct = get_metric('Control %')
    treatment_pct = get_metric('Treatment %')
    control_samples = get_metric('Control Samples')
    treatment_samples = get_metric('Treatment Samples')
    conversion_rate = get_metric('Conversion Rate %')
    ate = get_metric('ATE (percentage points)')
    relative_lift = get_metric('Relative Lift %')
    
    # Treatment vs Control comparison
    st.markdown('<p class="section-header">Treatment vs Control</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Treatment distribution
        fig_treatment = go.Figure()
        
        fig_treatment.add_trace(go.Bar(
            x=['Control', 'Treatment'],
            y=[control_samples, treatment_samples],
            marker_color=['#1565C0', '#2E7D32'],
            text=[f'{control_samples:,.0f}', f'{treatment_samples:,.0f}'],
            textposition='outside'
        ))
        
        fig_treatment.update_layout(
            title="Sample Distribution",
            xaxis_title="Group",
            yaxis_title="Number of Samples",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_treatment, use_container_width=True)
    
    with col2:
        # Distribution percentages
        fig_pct = go.Figure()
        
        fig_pct.add_trace(go.Bar(
            x=['Control', 'Treatment'],
            y=[control_pct, treatment_pct],
            marker_color=['#1565C0', '#2E7D32'],
            text=[f'{control_pct:.1f}%', f'{treatment_pct:.1f}%'],
            textposition='outside'
        ))
        
        fig_pct.update_layout(
            title="Treatment/Control Split",
            xaxis_title="Group",
            yaxis_title="Percentage (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_pct, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>Why 85% treatment?</strong> Unlike typical A/B tests where treatment is minimized 
    to reduce risk, this is an advertising dataset. Treatment (showing ads) generates revenue, 
    so the company wants to maximize it. The 15% control holdout is the minimum needed to 
    measure causal effects while maximizing ad revenue.
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown('<p class="section-header">Average Treatment Effect</p>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # Calculate approximate conversion rates (using ATE and overall conversion rate)
    # ATE = treatment_rate - control_rate
    # Overall rate is weighted average: 0.85 * treatment_rate + 0.15 * control_rate = conversion_rate
    # Solving: control_rate = (conversion_rate - 0.85 * ate) / (1 - 0.85 * (1 - 0.15/0.85))
    # Simplified: treatment_rate ≈ conversion_rate + 0.15 * ate, control_rate ≈ conversion_rate - 0.85 * ate
    control_conv_rate = conversion_rate - (treatment_pct/100) * ate
    treatment_conv_rate = conversion_rate + (control_pct/100) * ate
    
    with metric_col1:
        st.metric(
            "Control Conversion Rate",
            f"{control_conv_rate:.3f}%",
            help="Estimated conversion rate for users who did not see the ad"
        )
    
    with metric_col2:
        st.metric(
            "Treatment Conversion Rate",
            f"{treatment_conv_rate:.3f}%",
            help="Estimated conversion rate for users who saw the ad"
        )
    
    with metric_col3:
        st.metric(
            "Absolute ATE",
            f"+{ate:.3f}pp",
            help="Percentage point increase in conversion"
        )
    
    with metric_col4:
        st.metric(
            "Relative Lift",
            f"+{relative_lift:.1f}%",
            help="Percentage improvement over control"
        )
    
    st.markdown("""
    <div class="insight-box">
    <strong>My read on this:</strong> A ~60% relative lift sounds impressive, but keep in mind 
    we are talking about very low base rates here. Going from ~0.19% to ~0.31% is meaningful at 
    scale, but the absolute numbers are small. This is typical for digital advertising - 
    conversion rates are low, but with millions of impressions, even small improvements matter. 
    The real question is whether we can do better by being smarter about who we show ads to.
    </div>
    """, unsafe_allow_html=True)
    
    # Outcome distribution
    st.markdown('<p class="section-header">Outcome Distribution</p>', unsafe_allow_html=True)
    
    # Filter to get just the positive outcomes (value=1) for each outcome type
    positive_outcomes = outcome_dist[outcome_dist['value'] == 1].copy()
    
    fig_outcomes = go.Figure()
    
    fig_outcomes.add_trace(go.Bar(
        x=positive_outcomes['outcome'],
        y=positive_outcomes['percentage'],
        marker_color=['#C62828', '#1565C0', '#2E7D32'],
        text=positive_outcomes['percentage'].apply(lambda x: f'{x:.2f}%'),
        textposition='outside'
    ))
    
    fig_outcomes.update_layout(
        title="Outcome Rates Across All Users",
        xaxis_title="Outcome Type",
        yaxis_title="Rate (%)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_outcomes, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>Context:</strong> Conversion is the rarest outcome at ~0.3%, which makes sense - 
    this is the bottom of the funnel. Visit rate is around 4.7% and exposure around 3%. 
    The low conversion rate is actually what makes uplift modeling valuable here. If 
    everyone converted, there would be nothing to optimize. The scarcity is what creates 
    the opportunity.
    </div>
    """, unsafe_allow_html=True)
    
    # Heterogeneous effects
    st.markdown('<p class="section-header">Heterogeneous Treatment Effects</p>', unsafe_allow_html=True)
    
    st.markdown("""
    One of the key questions in uplift modeling is whether the treatment effect varies 
    across different user segments. If everyone responds the same way to ads, targeting 
    does not help. But if some users are more responsive than others, we can improve 
    efficiency by focusing on them.
    """)
    
    # Uplift by feature quartile
    if not uplift_quartile.empty:
        fig_quartile = go.Figure()
        
        fig_quartile.add_trace(go.Bar(
            x=uplift_quartile['Quartile'],
            y=uplift_quartile['Uplift'] * 100,
            marker_color=['#2E7D32', '#1565C0', '#616161'],
            text=uplift_quartile['Uplift'].apply(lambda x: f'{x*100:.3f}%'),
            textposition='outside'
        ))
        
        fig_quartile.add_hline(y=ate, line_dash="dash", line_color="red", 
                               annotation_text=f"Overall ATE: {ate:.3f}%")
        
        fig_quartile.update_layout(
            title="Uplift by Feature Quartile (f0)",
            xaxis_title="Quartile",
            yaxis_title="Observed Uplift (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_quartile, use_container_width=True)
        
        # Also show the underlying rates
        st.markdown("**Conversion Rates by Quartile:**")
        display_df = uplift_quartile.copy()
        display_df['Control_Rate'] = display_df['Control_Rate'].apply(lambda x: f'{x*100:.3f}%')
        display_df['Treatment_Rate'] = display_df['Treatment_Rate'].apply(lambda x: f'{x*100:.3f}%')
        display_df['Uplift'] = display_df['Uplift'].apply(lambda x: f'{x*100:.4f}%')
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>The takeaway:</strong> There is clear variation in treatment effects across 
    different feature values. Some user segments show much higher uplift than others. 
    This is exactly what uplift modeling tries to capture - identifying which users 
    are most responsive to treatment so we can target them more efficiently.
    </div>
    """, unsafe_allow_html=True)
    
    # Summary
    st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **What we know so far:**
    
    1. The advertising campaign works - there is a measurable positive effect on conversions
    2. The effect size is about +0.11 percentage points (roughly 60% relative lift)
    3. Treatment effects vary across user segments - some users are more responsive than others
    4. This heterogeneity is what makes uplift modeling valuable
    
    **Next step:** Look at Customer Segments to understand who the most responsive users are 
    and how to identify them.
    """)


if __name__ == "__main__":
    main()
