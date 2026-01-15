"""
ROI Calculator Page
Interactive tool to simulate targeting strategies and expected returns.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, COLORS
from utils.data_loader import load_predictions_with_segments, load_decile_analysis

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
    .result-box {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def calculate_targeting_metrics(predictions_df, target_pct):
    """Calculate expected metrics for a given targeting percentage."""
    # Sort by predicted uplift (descending)
    sorted_df = predictions_df.sort_values('uplift_pred', ascending=False)
    
    n_total = len(sorted_df)
    n_target = int(n_total * target_pct / 100)
    
    # Get targeted and non-targeted groups
    targeted = sorted_df.head(n_target)
    non_targeted = sorted_df.tail(n_total - n_target)
    
    # Calculate metrics for targeted group
    targeted_uplift = targeted['uplift_pred'].mean()
    
    # Calculate actual conversion rates (if we have ground truth)
    targeted_treated = targeted[targeted['treatment'] == 1]
    targeted_control = targeted[targeted['treatment'] == 0]
    
    if len(targeted_treated) > 0 and len(targeted_control) > 0:
        actual_treated_rate = targeted_treated['y_true'].mean()
        actual_control_rate = targeted_control['y_true'].mean()
        actual_uplift = actual_treated_rate - actual_control_rate
    else:
        actual_uplift = targeted_uplift  # Use predicted if we can't calculate actual
    
    return {
        'n_targeted': n_target,
        'predicted_uplift': targeted_uplift,
        'actual_uplift': actual_uplift,
        'targeted_segment_dist': targeted['segment'].value_counts().to_dict() if 'segment' in targeted.columns else {}
    }


def main():
    st.title("ROI Calculator")
    st.markdown("*Interactive tool to simulate targeting strategies and expected returns*")
    
    # Load data
    predictions = load_predictions_with_segments()
    decile_data = load_decile_analysis()
    t_learner_deciles = decile_data[decile_data['Model'] == 'T-Learner']
    
    # Introduction
    st.markdown("""
    <div class="insight-box">
    <strong>How to use this calculator:</strong> Adjust the targeting percentage slider 
    to see how many customers you would reach and what the expected uplift would be. 
    The tool helps answer questions like "What if we only target the top 10% of customers 
    by predicted uplift?"
    </div>
    """, unsafe_allow_html=True)
    
    # Targeting simulator
    st.markdown('<p class="section-header">Targeting Simulator</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_pct = st.slider(
            "Target Top % of Customers",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="Select what percentage of customers to target, sorted by predicted uplift"
        )
        
        # Input parameters for ROI calculation
        st.markdown("**Campaign Parameters:**")
        
        total_customers = st.number_input(
            "Total Customer Base",
            min_value=1000,
            max_value=10000000,
            value=1000000,
            step=10000,
            help="Size of your customer base"
        )
        
        cost_per_impression = st.number_input(
            "Cost per Impression ($)",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Cost to show one ad"
        )
        
        revenue_per_conversion = st.number_input(
            "Revenue per Conversion ($)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=5.0,
            help="Average revenue from one conversion"
        )
    
    with col2:
        # Calculate metrics
        metrics = calculate_targeting_metrics(predictions, target_pct)
        
        # Scale to user's customer base
        scale_factor = total_customers / len(predictions)
        n_targeted_scaled = int(metrics['n_targeted'] * scale_factor)
        
        # Calculate financials
        ad_spend = n_targeted_scaled * cost_per_impression
        incremental_conversions = n_targeted_scaled * metrics['actual_uplift']
        incremental_revenue = incremental_conversions * revenue_per_conversion
        roi = ((incremental_revenue - ad_spend) / ad_spend * 100) if ad_spend > 0 else 0
        
        # Display results
        st.markdown("**Projected Results:**")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(
                "Customers Targeted",
                f"{n_targeted_scaled:,}",
                delta=f"{target_pct}% of base"
            )
            st.metric(
                "Ad Spend",
                f"${ad_spend:,.0f}",
                help="Total cost of impressions"
            )
        
        with result_col2:
            st.metric(
                "Incremental Conversions",
                f"{incremental_conversions:,.0f}",
                delta=f"Uplift: {metrics['actual_uplift']*100:.3f}%"
            )
            st.metric(
                "Incremental Revenue",
                f"${incremental_revenue:,.0f}",
                help="Revenue from incremental conversions only"
            )
        
        # ROI highlight
        roi_color = "#2E7D32" if roi > 0 else "#C62828"
        st.markdown(f"""
        <div class="result-box">
            <div style="font-size: 1rem; opacity: 0.9;">Estimated ROI</div>
            <div style="font-size: 2.5rem; font-weight: 700;">{roi:+.0f}%</div>
            <div style="font-size: 0.9rem; opacity: 0.8;">Net: ${incremental_revenue - ad_spend:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Segment composition
    st.markdown('<p class="section-header">Who Are We Targeting?</p>', unsafe_allow_html=True)
    
    if metrics['targeted_segment_dist']:
        segment_order = ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']
        segment_colors = {
            'Persuadables': '#2E7D32',
            'Sure Things': '#1565C0',
            'Lost Causes': '#616161',
            'Sleeping Dogs': '#C62828'
        }
        
        fig_segments = go.Figure()
        
        for segment in segment_order:
            count = metrics['targeted_segment_dist'].get(segment, 0)
            pct = count / metrics['n_targeted'] * 100 if metrics['n_targeted'] > 0 else 0
            
            fig_segments.add_trace(go.Bar(
                x=[segment],
                y=[pct],
                name=segment,
                marker_color=segment_colors[segment],
                text=[f'{pct:.1f}%'],
                textposition='outside'
            ))
        
        fig_segments.update_layout(
            title=f"Segment Composition of Top {target_pct}% Targeted Customers",
            xaxis_title="Segment",
            yaxis_title="Percentage of Targeted Group",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_segments, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>What this tells us:</strong> As you increase the targeting percentage, the 
        composition of your targeted group changes. At lower percentages (say 10-25%), you 
        are mostly reaching Persuadables - the high-value customers. As you expand targeting, 
        you start including more Sure Things, Lost Causes, and eventually Sleeping Dogs, 
        which dilutes the average uplift.
        </div>
        """, unsafe_allow_html=True)
    
    # Efficiency curve
    st.markdown('<p class="section-header">Targeting Efficiency Curve</p>', unsafe_allow_html=True)
    
    # Calculate metrics for different targeting levels
    efficiency_data = []
    for pct in range(5, 101, 5):
        m = calculate_targeting_metrics(predictions, pct)
        n_scaled = int(m['n_targeted'] * scale_factor)
        spend = n_scaled * cost_per_impression
        inc_conv = n_scaled * m['actual_uplift']
        inc_rev = inc_conv * revenue_per_conversion
        calc_roi = ((inc_rev - spend) / spend * 100) if spend > 0 else 0
        
        efficiency_data.append({
            'Target %': pct,
            'Customers': n_scaled,
            'Predicted Uplift': m['predicted_uplift'] * 100,
            'Actual Uplift': m['actual_uplift'] * 100,
            'ROI': calc_roi,
            'Net Revenue': inc_rev - spend
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    # Plot efficiency curves
    fig_efficiency = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Uplift by Targeting Level', 'ROI by Targeting Level')
    )
    
    fig_efficiency.add_trace(
        go.Scatter(
            x=efficiency_df['Target %'],
            y=efficiency_df['Actual Uplift'],
            mode='lines+markers',
            name='Actual Uplift',
            line=dict(color='#2E7D32', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    fig_efficiency.add_trace(
        go.Scatter(
            x=efficiency_df['Target %'],
            y=efficiency_df['ROI'],
            mode='lines+markers',
            name='ROI',
            line=dict(color='#1565C0', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )
    
    # Add reference line for current selection
    fig_efficiency.add_vline(x=target_pct, line_dash="dash", line_color="red", row=1, col=1)
    fig_efficiency.add_vline(x=target_pct, line_dash="dash", line_color="red", row=1, col=2)
    
    fig_efficiency.update_layout(
        height=400,
        showlegend=False
    )
    
    fig_efficiency.update_xaxes(title_text="Target %", row=1, col=1)
    fig_efficiency.update_xaxes(title_text="Target %", row=1, col=2)
    fig_efficiency.update_yaxes(title_text="Uplift (%)", row=1, col=1)
    fig_efficiency.update_yaxes(title_text="ROI (%)", row=1, col=2)
    
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>The efficiency trade-off:</strong> Both uplift and ROI decrease as you expand 
    targeting. The highest uplift comes from targeting only the top 10% (mostly Persuadables), 
    but this limits your total reach. There is a sweet spot where you balance efficiency 
    with scale - typically somewhere in the 20-30% range for this data. The red dashed line 
    shows your current selection.
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    st.markdown('<p class="section-header">Scenario Comparison</p>', unsafe_allow_html=True)
    
    comparison_pcts = [10, 25, 50, 100]
    comparison_data = []
    
    for pct in comparison_pcts:
        m = calculate_targeting_metrics(predictions, pct)
        n_scaled = int(m['n_targeted'] * scale_factor)
        spend = n_scaled * cost_per_impression
        inc_conv = n_scaled * m['actual_uplift']
        inc_rev = inc_conv * revenue_per_conversion
        calc_roi = ((inc_rev - spend) / spend * 100) if spend > 0 else 0
        
        comparison_data.append({
            'Scenario': f'Target Top {pct}%',
            'Customers': f'{n_scaled:,}',
            'Ad Spend': f'${spend:,.0f}',
            'Incr. Conversions': f'{inc_conv:,.0f}',
            'Incr. Revenue': f'${inc_rev:,.0f}',
            'ROI': f'{calc_roi:+.0f}%'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    # My take
    st.markdown('<p class="section-header">My Take on Targeting Strategy</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Looking at these numbers, here is how I would think about targeting strategy:
    
    **The case for narrow targeting (10-20%):** If your primary goal is efficiency and 
    you have limited budget, targeting only the top 10-20% makes sense. You get the 
    highest uplift per impression and the best ROI. The downside is limited reach - 
    you are leaving potential conversions on the table.
    
    **The case for broader targeting (30-50%):** If you have budget to spend and want 
    to maximize total incremental conversions, broader targeting captures more volume. 
    The per-impression efficiency drops, but the absolute number of conversions increases. 
    This makes sense if revenue growth is the primary objective.
    
    **What I would avoid:** Targeting 100% of customers is almost certainly a mistake. 
    The bottom 25-30% includes Sleeping Dogs where ads actively hurt conversion. Even 
    the middle segments are borderline - you are paying for impressions that generate 
    little to no incremental value.
    
    **My recommendation:** Start with the top 20-25% and measure results. If you have 
    budget headroom and see good performance, gradually expand. Use the ROI curve as 
    a guide - when ROI drops below your threshold, you have found your optimal reach.
    
    **Important caveat:** These projections assume the model's predictions hold in 
    production. I would recommend a holdout test to validate before scaling up spend.
    """)


if __name__ == "__main__":
    main()


