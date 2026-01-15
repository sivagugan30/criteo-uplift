"""
Uplift Modeling Dashboard - Homepage
Executive summary and navigation for business stakeholders.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import PAGE_CONFIG, COLORS, SEGMENTS
from utils.data_loader import (
    load_model_metrics,
    load_customer_segments,
    load_eda_summary,
    load_decile_analysis
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
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
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">Uplift Modeling Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Criteo Advertising Campaign Analysis</p>', unsafe_allow_html=True)
    
    # Load key data
    model_metrics = load_model_metrics()
    segments = load_customer_segments()
    eda_summary = load_eda_summary()
    decile_data = load_decile_analysis()
    
    # Key metrics row
    st.markdown('<p class="section-header">Key Metrics at a Glance</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate key metrics
    best_model = model_metrics.loc[model_metrics['Qini_Coefficient'].idxmax()]
    persuadables_pct = segments[segments['Segment'] == 'Persuadables']['Percentage'].values[0]
    
    # Top decile lift (T-Learner)
    t_learner_decile = decile_data[decile_data['Model'] == 'T-Learner']
    top_decile_lift = t_learner_decile[t_learner_decile['Decile'] == 1]['Lift vs Random'].values[0]
    
    # Get relative lift from EDA summary
    relative_lift = eda_summary[eda_summary['Metric'] == 'Relative Lift %']['Value'].values[0]
    
    with col1:
        st.metric(
            label="Best Model",
            value=best_model['Model'],
            delta=f"Qini: {best_model['Qini_Coefficient']:.1f}"
        )
    
    with col2:
        st.metric(
            label="Persuadables",
            value=f"{persuadables_pct:.1f}%",
            delta="of customer base"
        )
    
    with col3:
        st.metric(
            label="Top Decile Lift",
            value=f"{top_decile_lift:.1f}x",
            delta="vs random targeting"
        )
    
    with col4:
        st.metric(
            label="Campaign Lift",
            value=f"+{relative_lift:.0f}%",
            delta="conversion improvement"
        )
    
    # Executive summary
    st.markdown('<p class="section-header">Executive Summary</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>The Bottom Line:</strong> This analysis of the Criteo advertising dataset shows that 
    targeted advertising works, but not equally for everyone. About 25% of our customer base 
    are "Persuadables" who convert specifically because they saw the ad. By focusing our 
    ad spend on this group, we can achieve 7x better results than random targeting.
    </div>
    """, unsafe_allow_html=True)
    
    # What we found
    st.markdown("**What the data tells us:**")
    
    findings_col1, findings_col2 = st.columns(2)
    
    with findings_col1:
        st.markdown("""
        - The advertising campaign has a measurable positive effect on conversions
        - Not all customers respond the same way to ads
        - Some customers would buy anyway (we can save money here)
        - A small subset actually converts less when shown ads
        """)
    
    with findings_col2:
        st.markdown("""
        - The T-Learner model best identifies who to target
        - Feature f2 and f8 are the strongest predictors of uplift
        - Targeting the top 10% yields the highest ROI
        - Model predictions align well with actual observed behavior
        """)
    
    # Navigation guide
    st.markdown('<p class="section-header">Dashboard Guide</p>', unsafe_allow_html=True)
    
    pages_data = {
        "Page": [
            "Campaign Overview",
            "Customer Segments",
            "Model Performance",
            "Customer Lookup",
            "ROI Calculator",
            "Technical Details"
        ],
        "Description": [
            "Treatment effect analysis and overall campaign performance metrics",
            "Four customer types and recommended actions for each segment",
            "Model comparison, Qini curves, and why T-Learner performs best",
            "Search individual customers and understand their segment assignment",
            "Interactive tool to simulate targeting strategies and expected returns",
            "Methodology, model specifications, and statistical details for data science team"
        ]
    }
    
    st.dataframe(
        pd.DataFrame(pages_data),
        hide_index=True,
        use_container_width=True
    )
    
    # Quick insight
    st.markdown('<p class="section-header">My Take on This</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Looking at this data, I think the most actionable insight is the customer segmentation. 
    The math suggests that about a quarter of the customer base are "Persuadables" - people 
    who genuinely change their behavior because of the ad. Another quarter are "Sleeping Dogs" 
    where showing ads actually hurts conversion rates, which is counterintuitive but the data 
    is pretty clear on this.
    
    If I had to pick one thing to act on, it would be this: stop treating everyone the same. 
    The difference between targeting Persuadables versus random targeting is roughly 7x in terms 
    of incremental conversions. That is a significant efficiency gain that should translate 
    directly to ad spend ROI.
    
    The T-Learner model came out on top here, which is interesting because the more complex 
    X-Learner was specifically designed for situations with imbalanced treatment/control splits 
    like this one. But with 45,000+ samples in the control group, the simpler approach won. 
    Sometimes simpler is better.
    """)
    
    # Footer
    st.divider()
    st.markdown(
        "*Dashboard built on Criteo Uplift Dataset | "
        "Use the sidebar to navigate between pages*",
        help="Data source: Criteo/Hugging Face"
    )


if __name__ == "__main__":
    main()
