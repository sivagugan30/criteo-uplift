"""
Customer Lookup Page
Search individual customers and understand their segment assignment.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, COLORS, SEGMENTS, FEATURE_NAMES
from utils.data_loader import load_predictions_with_segments, load_shap_importance

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
    .customer-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .persuadables-card { border-left: 4px solid #2E7D32; }
    .sure-things-card { border-left: 4px solid #1565C0; }
    .lost-causes-card { border-left: 4px solid #616161; }
    .sleeping-dogs-card { border-left: 4px solid #C62828; }
</style>
""", unsafe_allow_html=True)

# Segment colors
SEGMENT_COLORS = {
    'Persuadables': '#2E7D32',
    'Sure Things': '#1565C0',
    'Lost Causes': '#616161',
    'Sleeping Dogs': '#C62828'
}

SEGMENT_ACTIONS = {
    'Persuadables': 'Target - ads drive incremental conversions',
    'Sure Things': 'Save budget - would convert anyway',
    'Lost Causes': 'Skip - will not convert regardless',
    'Sleeping Dogs': 'Avoid - ads hurt conversion'
}


@st.cache_data
def get_sample_customers(predictions_df, n_per_segment=5):
    """Get sample customers from each segment for browsing."""
    samples = []
    for segment in ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']:
        seg_data = predictions_df[predictions_df['segment'] == segment].head(n_per_segment)
        samples.append(seg_data)
    return pd.concat(samples).reset_index(drop=True)


def display_customer_profile(customer_row, shap_importance):
    """Display detailed profile for a single customer."""
    segment = customer_row['segment']
    segment_class = segment.lower().replace(' ', '-')
    
    # Customer card
    st.markdown(f"""
    <div class="customer-card {segment_class}-card">
        <h3>Customer ID: {customer_row.name}</h3>
        <p><strong>Segment:</strong> {segment}</p>
        <p><strong>Action:</strong> {SEGMENT_ACTIONS[segment]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Predicted Uplift",
            f"{customer_row['uplift_pred'] * 100:.4f}%",
            help="Model's prediction of how much the ad would change conversion probability"
        )
    
    with col2:
        if 'baseline_prob' in customer_row:
            st.metric(
                "Baseline Probability",
                f"{customer_row['baseline_prob'] * 100:.4f}%",
                help="Probability of conversion without seeing the ad"
            )
    
    with col3:
        if 'treatment_prob' in customer_row:
            st.metric(
                "Treatment Probability",
                f"{customer_row['treatment_prob'] * 100:.4f}%",
                help="Probability of conversion after seeing the ad"
            )
    
    with col4:
        st.metric(
            "Actual Outcome",
            "Converted" if customer_row['y_true'] == 1 else "Did not convert",
            delta="Treated" if customer_row['treatment'] == 1 else "Control"
        )
    
    # Segment explanation
    st.markdown('<p class="section-header">What This Means</p>', unsafe_allow_html=True)
    
    explanations = {
        'Persuadables': """
        This customer has a positive predicted uplift, meaning the model believes 
        showing them an ad would increase their probability of conversion. These are 
        the customers where ad spend generates real incremental value. The recommendation 
        is to target this customer with advertising.
        """,
        'Sure Things': """
        This customer has a high baseline conversion probability but low uplift. In 
        other words, they were likely to convert anyway, with or without the ad. Showing 
        them an ad does not hurt, but it also does not help much. The recommendation is 
        to save ad budget and deprioritize this customer.
        """,
        'Lost Causes': """
        This customer has both low baseline probability and low uplift. They are unlikely 
        to convert regardless of whether they see an ad. Targeting them would waste 
        ad spend without generating incremental conversions. The recommendation is to 
        skip this customer entirely.
        """,
        'Sleeping Dogs': """
        This customer has negative predicted uplift, meaning the model believes showing 
        them an ad would actually decrease their probability of conversion. This is 
        counterintuitive but real - some customers react negatively to advertising. 
        The recommendation is to actively avoid targeting this customer.
        """
    }
    
    st.markdown(f"""
    <div class="insight-box">
    {explanations[segment]}
    </div>
    """, unsafe_allow_html=True)
    
    # Feature values
    st.markdown('<p class="section-header">Feature Profile</p>', unsafe_allow_html=True)
    
    # Get feature values
    feature_data = []
    for feat in FEATURE_NAMES:
        if feat in customer_row:
            feature_data.append({
                'Feature': feat,
                'Value': customer_row[feat],
                'Importance': shap_importance[shap_importance['Feature'] == feat]['Importance'].values[0] if feat in shap_importance['Feature'].values else 0
            })
    
    feature_df = pd.DataFrame(feature_data)
    feature_df = feature_df.sort_values('Importance', ascending=False)
    
    # Feature bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=feature_df['Feature'],
        y=feature_df['Value'],
        marker_color='#1565C0',
        text=feature_df['Value'].apply(lambda x: f'{x:.2f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Values (Sorted by Global Importance)",
        xaxis_title="Feature",
        yaxis_title="Value",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature table
    feature_display = feature_df.copy()
    feature_display['Value'] = feature_display['Value'].apply(lambda x: f'{x:.4f}')
    feature_display['Importance'] = feature_display['Importance'].apply(lambda x: f'{x:.6f}')
    
    st.dataframe(feature_display, hide_index=True, use_container_width=True)


def main():
    st.title("Customer Lookup")
    st.markdown("*Search individual customers and understand their segment assignment*")
    
    # Load data
    predictions = load_predictions_with_segments()
    shap_importance = load_shap_importance()
    
    # Introduction
    st.markdown("""
    <div class="insight-box">
    <strong>How to use this page:</strong> Enter a customer ID to see their profile, 
    predicted uplift, and segment assignment. You can also browse sample customers from 
    each segment to understand what typical profiles look like in each category.
    </div>
    """, unsafe_allow_html=True)
    
    # Search interface
    st.markdown('<p class="section-header">Customer Search</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        customer_id = st.number_input(
            "Enter Customer ID",
            min_value=0,
            max_value=len(predictions) - 1,
            value=0,
            step=1,
            help=f"Valid IDs: 0 to {len(predictions) - 1}"
        )
    
    with col2:
        if st.button("Random Customer"):
            customer_id = np.random.randint(0, len(predictions))
            st.rerun()
    
    # Display selected customer
    if customer_id is not None:
        customer = predictions.iloc[customer_id]
        display_customer_profile(customer, shap_importance)
    
    # Segment browser
    st.markdown('<p class="section-header">Browse by Segment</p>', unsafe_allow_html=True)
    
    selected_segment = st.selectbox(
        "Select a segment to browse sample customers",
        options=['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']
    )
    
    # Get samples from selected segment
    segment_samples = predictions[predictions['segment'] == selected_segment].head(20)
    
    # Display summary table
    display_cols = ['segment', 'uplift_pred', 'y_true', 'treatment']
    if 'baseline_prob' in segment_samples.columns:
        display_cols.insert(2, 'baseline_prob')
    
    display_df = segment_samples[display_cols].copy()
    display_df['uplift_pred'] = display_df['uplift_pred'].apply(lambda x: f'{x*100:.4f}%')
    if 'baseline_prob' in display_df.columns:
        display_df['baseline_prob'] = display_df['baseline_prob'].apply(lambda x: f'{x*100:.4f}%')
    display_df['y_true'] = display_df['y_true'].apply(lambda x: 'Yes' if x == 1 else 'No')
    display_df['treatment'] = display_df['treatment'].apply(lambda x: 'Treated' if x == 1 else 'Control')
    
    display_df.columns = ['Segment', 'Predicted Uplift', 'Baseline Prob', 'Converted', 'Group'] if 'baseline_prob' in display_cols else ['Segment', 'Predicted Uplift', 'Converted', 'Group']
    
    st.dataframe(display_df, hide_index=False, use_container_width=True)
    
    st.markdown("""
    *Click on any row's index number in the Customer Search box above to view their full profile.*
    """)
    
    # Segment statistics
    st.markdown('<p class="section-header">Segment Summary Statistics</p>', unsafe_allow_html=True)
    
    segment_stats = predictions.groupby('segment').agg({
        'uplift_pred': ['mean', 'std', 'min', 'max'],
        'y_true': 'mean'
    }).round(6)
    
    segment_stats.columns = ['Mean Uplift', 'Std Uplift', 'Min Uplift', 'Max Uplift', 'Conversion Rate']
    segment_stats = segment_stats.reset_index()
    segment_stats = segment_stats[segment_stats['segment'].isin(['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs'])]
    
    # Format for display
    for col in ['Mean Uplift', 'Std Uplift', 'Min Uplift', 'Max Uplift']:
        segment_stats[col] = segment_stats[col].apply(lambda x: f'{x*100:.4f}%')
    segment_stats['Conversion Rate'] = segment_stats['Conversion Rate'].apply(lambda x: f'{x*100:.2f}%')
    segment_stats.columns = ['Segment', 'Mean Uplift', 'Std Dev', 'Min Uplift', 'Max Uplift', 'Conversion Rate']
    
    st.dataframe(segment_stats, hide_index=True, use_container_width=True)
    
    # My take
    st.markdown('<p class="section-header">Interpreting Individual Profiles</p>', unsafe_allow_html=True)
    
    st.markdown("""
    When looking at individual customer profiles, keep a few things in mind:
    
    **On the features:** The features in this dataset are anonymized (f0, f1, etc.), so we 
    cannot interpret them directly. In a real-world application, you would see meaningful 
    attributes like demographics, purchase history, or engagement metrics. The important 
    thing is that the model uses these features to predict who will respond to treatment.
    
    **On the predictions:** The uplift values are small in absolute terms (typically less 
    than 1%). This is normal for digital advertising where base conversion rates are low. 
    What matters is the relative difference between segments - Persuadables have meaningfully 
    higher uplift than other segments.
    
    **On uncertainty:** The model gives point predictions, but there is always uncertainty. 
    A customer labeled as a Persuadable is not guaranteed to respond positively to ads - 
    it just means the model estimates a higher probability of response compared to other 
    customers. At scale, these probabilities translate to real differences in outcomes.
    
    **On individual vs. aggregate decisions:** Looking at individual customers is useful 
    for understanding the model, but business decisions should be made at the aggregate 
    level. Targeting all Persuadables will yield better results than random targeting, 
    even though individual outcomes will vary.
    """)


if __name__ == "__main__":
    main()


