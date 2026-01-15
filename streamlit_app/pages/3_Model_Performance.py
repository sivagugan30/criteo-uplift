"""
Model Performance Page
Model comparison, Qini curves, and why T-Learner performs best.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, COLORS
from utils.data_loader import (
    load_model_metrics,
    load_qini_curves,
    load_decile_analysis,
    load_predictions
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

# Model colors
MODEL_COLORS = {
    'T-Learner': '#2E7D32',
    'S-Learner': '#1565C0',
    'X-Learner': '#FF6F00'
}


def main():
    st.title("Model Performance")
    st.markdown("*Model comparison, Qini curves, and why T-Learner performs best*")
    
    # Load data
    model_metrics = load_model_metrics()
    qini_curves = load_qini_curves()
    decile_data = load_decile_analysis()
    
    # Introduction
    st.markdown("""
    <div class="insight-box">
    <strong>Why model selection matters:</strong> Different uplift models make different 
    trade-offs. The goal is to find the model that best ranks users by their true 
    treatment responsiveness. A better ranking means more efficient targeting and 
    higher ROI on ad spend.
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison
    st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
    
    # Sort by Qini coefficient
    metrics_sorted = model_metrics.sort_values('Qini_Coefficient', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Qini coefficient bar chart
        fig_qini = go.Figure()
        
        fig_qini.add_trace(go.Bar(
            x=metrics_sorted['Model'],
            y=metrics_sorted['Qini_Coefficient'],
            marker_color=[MODEL_COLORS[m] for m in metrics_sorted['Model']],
            text=metrics_sorted['Qini_Coefficient'].apply(lambda x: f'{x:.1f}'),
            textposition='outside'
        ))
        
        fig_qini.update_layout(
            title="Qini Coefficient by Model",
            xaxis_title="Model",
            yaxis_title="Qini Coefficient",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_qini, use_container_width=True)
    
    with col2:
        # AUUC bar chart
        fig_auuc = go.Figure()
        
        fig_auuc.add_trace(go.Bar(
            x=metrics_sorted['Model'],
            y=metrics_sorted['AUUC'] * 1000,  # Scale for readability
            marker_color=[MODEL_COLORS[m] for m in metrics_sorted['Model']],
            text=metrics_sorted['AUUC'].apply(lambda x: f'{x*1000:.2f}'),
            textposition='outside'
        ))
        
        fig_auuc.update_layout(
            title="AUUC by Model (x1000)",
            xaxis_title="Model",
            yaxis_title="AUUC (x1000)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_auuc, use_container_width=True)
    
    # Metrics table
    st.markdown("**Performance Metrics:**")
    
    display_metrics = metrics_sorted.copy()
    display_metrics['Qini_Coefficient'] = display_metrics['Qini_Coefficient'].apply(lambda x: f'{x:.2f}')
    display_metrics['AUUC'] = display_metrics['AUUC'].apply(lambda x: f'{x:.6f}')
    display_metrics['Mean_Predicted_Uplift'] = display_metrics['Mean_Predicted_Uplift'].apply(lambda x: f'{x*100:.4f}%')
    display_metrics.columns = ['Model', 'Qini Coefficient', 'AUUC', 'Mean Predicted Uplift']
    
    st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>What these metrics mean:</strong>
    <br><br>
    <strong>Qini Coefficient:</strong> Measures how well the model ranks users by uplift. 
    Higher is better. It is the area between the model's curve and random targeting.
    <br><br>
    <strong>AUUC (Area Under Uplift Curve):</strong> Average uplift across all targeting 
    thresholds. Higher is better.
    <br><br>
    T-Learner wins on both metrics, with a Qini coefficient almost double the S-Learner 
    and more than 5x the X-Learner.
    </div>
    """, unsafe_allow_html=True)
    
    # Qini curves
    st.markdown('<p class="section-header">Qini Curves</p>', unsafe_allow_html=True)
    
    st.markdown("""
    The Qini curve shows cumulative incremental conversions as we target more users, 
    starting with those predicted to have the highest uplift. A curve that rises steeply 
    and stays above the diagonal (random targeting) indicates a model that successfully 
    identifies high-uplift users.
    """)
    
    fig_curves = go.Figure()
    
    # Note: column names are lowercase in the CSV
    for model in ['T-Learner', 'S-Learner', 'X-Learner']:
        model_data = qini_curves[qini_curves['model'] == model]
        fig_curves.add_trace(go.Scatter(
            x=model_data['percentile'],
            y=model_data['qini'],
            mode='lines',
            name=model,
            line=dict(color=MODEL_COLORS[model], width=2)
        ))
    
    # Add random baseline
    max_qini = qini_curves['qini'].max()
    fig_curves.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, max_qini],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig_curves.update_layout(
        title="Qini Curves - Model Comparison",
        xaxis_title="Fraction of Population Targeted",
        yaxis_title="Cumulative Incremental Conversions",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_curves, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>Reading the curves:</strong> The T-Learner curve (green) rises most steeply, 
    meaning it does the best job of putting high-uplift users first. Notice how all three 
    models eventually converge as we approach 100% of the population - at that point, 
    targeting makes no difference because everyone is included. The value of uplift 
    modeling shows up in the left portion of the curve, where selective targeting matters.
    </div>
    """, unsafe_allow_html=True)
    
    # Decile analysis
    st.markdown('<p class="section-header">Decile Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Breaking down performance by decile helps us understand where each model adds value. 
    Ideally, the top decile (users the model thinks have highest uplift) should have the 
    highest actual uplift.
    """)
    
    # T-Learner decile analysis
    t_learner_deciles = decile_data[decile_data['Model'] == 'T-Learner'].copy()
    
    fig_decile = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Predicted vs Actual Uplift', 'Lift vs Random Targeting')
    )
    
    fig_decile.add_trace(
        go.Bar(
            x=t_learner_deciles['Decile'],
            y=t_learner_deciles['Avg Predicted Uplift (%)'],
            name='Predicted',
            marker_color='#1565C0'
        ),
        row=1, col=1
    )
    
    fig_decile.add_trace(
        go.Scatter(
            x=t_learner_deciles['Decile'],
            y=t_learner_deciles['Actual Uplift (%)'],
            name='Actual',
            mode='lines+markers',
            line=dict(color='#C62828', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    fig_decile.add_trace(
        go.Bar(
            x=t_learner_deciles['Decile'],
            y=t_learner_deciles['Lift vs Random'],
            name='Lift',
            marker_color=t_learner_deciles['Lift vs Random'].apply(
                lambda x: '#2E7D32' if x > 0 else '#C62828'
            ).tolist()
        ),
        row=1, col=2
    )
    
    fig_decile.update_layout(
        title="T-Learner Performance by Decile",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5)
    )
    
    fig_decile.update_xaxes(title_text="Decile", row=1, col=1)
    fig_decile.update_xaxes(title_text="Decile", row=1, col=2)
    fig_decile.update_yaxes(title_text="Uplift (%)", row=1, col=1)
    fig_decile.update_yaxes(title_text="Lift vs Random", row=1, col=2)
    
    st.plotly_chart(fig_decile, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>What I see here:</strong> The T-Learner does a good job in the top decile - 
    users it predicts have the highest uplift actually do have the highest actual uplift 
    (around 0.8%). The lift versus random targeting in decile 1 is about 7x, which is 
    substantial. The model is doing what we want it to do: identifying the users who are 
    most responsive to treatment.
    <br><br>
    The middle deciles are noisier, which is expected - these are users the model is 
    less certain about. The bottom deciles have near-zero or negative uplift, which is 
    also what we want to see.
    </div>
    """, unsafe_allow_html=True)
    
    # Why T-Learner won
    st.markdown('<p class="section-header">Why T-Learner Performed Best</p>', unsafe_allow_html=True)
    
    st.markdown("""
    The surprising result here is that T-Learner, the simplest of the three models, 
    outperformed the more sophisticated X-Learner. Here is my thinking on why:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **T-Learner approach:**
        - Train one model on control group
        - Train another model on treatment group
        - Uplift = difference in predictions
        
        Pros: Simple, interpretable, fewer moving parts
        
        Cons: Requires sufficient samples in both groups
        """)
    
    with col2:
        st.markdown("""
        **X-Learner approach:**
        - Four-stage process with counterfactual imputation
        - Designed for imbalanced treatment/control splits
        - Uses propensity weighting to blend estimates
        
        Pros: Theoretically optimal for small control groups
        
        Cons: Complex, more opportunities for error propagation
        """)
    
    st.markdown("""
    <div class="insight-box">
    <strong>My hypothesis:</strong> X-Learner was designed for scenarios where the control 
    group is very small (say, less than 5% or fewer than 10k samples). In this dataset, 
    we have about 45,000 control samples, which is plenty for T-Learner to work well. 
    The additional complexity of X-Learner introduces more variance without providing 
    enough bias reduction to compensate.
    <br><br>
    This is a good reminder that more complex models are not always better. The right 
    model depends on the data. With sufficient sample sizes, simpler approaches often win.
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison table
    st.markdown("**Model Characteristics:**")
    
    model_comparison = pd.DataFrame({
        'Model': ['T-Learner', 'S-Learner', 'X-Learner'],
        'Complexity': ['Low', 'Low', 'High'],
        'Training Stages': ['2', '1', '4'],
        'Best For': [
            'Sufficient samples in both groups',
            'Simple baseline, interpretable',
            'Very imbalanced treatment/control'
        ],
        'Our Result': ['Winner', '2nd place', '3rd place']
    })
    
    st.dataframe(model_comparison, hide_index=True, use_container_width=True)
    
    # Summary
    st.markdown('<p class="section-header">Summary</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Key takeaways:**
    
    1. **T-Learner is the best model for this data** - Qini coefficient of 35.2, nearly 
       double the S-Learner and 5x the X-Learner
    
    2. **Top decile targeting yields 7x lift** - Users the model identifies as highest-uplift 
       actually are, with 7x better conversion lift than random targeting
    
    3. **Simpler can be better** - The more complex X-Learner underperformed because we 
       have enough data for the simpler T-Learner to work well
    
    4. **Recommendation:** Deploy T-Learner for production uplift scoring. It is simple, 
       effective, and well-validated on this data.
    """)


if __name__ == "__main__":
    main()
