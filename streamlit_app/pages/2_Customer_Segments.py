"""
Customer Segments Page
Four customer types and recommended actions for each segment.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, COLORS, SEGMENTS
from utils.data_loader import load_customer_segments, load_predictions_with_segments

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
    .segment-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .persuadables { background-color: #E8F5E9; border-left: 4px solid #2E7D32; }
    .sure-things { background-color: #E3F2FD; border-left: 4px solid #1565C0; }
    .lost-causes { background-color: #FAFAFA; border-left: 4px solid #616161; }
    .sleeping-dogs { background-color: #FFEBEE; border-left: 4px solid #C62828; }
</style>
""", unsafe_allow_html=True)

# Segment colors for charts
SEGMENT_COLORS = {
    'Persuadables': '#2E7D32',
    'Sure Things': '#1565C0',
    'Lost Causes': '#616161',
    'Sleeping Dogs': '#C62828'
}


def main():
    st.title("Customer Segments")
    st.markdown("*Four customer types and recommended actions for each segment*")
    
    # Load data
    segments = load_customer_segments()
    predictions = load_predictions_with_segments()
    
    # Introduction
    st.markdown("""
    <div class="insight-box">
    <strong>The core idea:</strong> Not all customers respond to advertising the same way. 
    Uplift modeling identifies four distinct customer types based on how their behavior 
    changes (or does not change) when shown an ad. Understanding these segments is the 
    foundation for efficient ad targeting.
    </div>
    """, unsafe_allow_html=True)
    
    # The four segments explained
    st.markdown('<p class="section-header">The Four Customer Types</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="segment-card persuadables">
        <strong>Persuadables</strong><br>
        <em>Target these customers</em><br><br>
        These users convert specifically because they see the ad. Without the ad, they would 
        not have converted. This is where ad spend creates real value. Every dollar spent 
        reaching Persuadables generates incremental revenue.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="segment-card lost-causes">
        <strong>Lost Causes</strong><br>
        <em>Skip these customers</em><br><br>
        These users will not convert regardless of whether they see an ad or not. Showing 
        them ads costs money but generates no incremental revenue. Resources spent here 
        are wasted.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="segment-card sure-things">
        <strong>Sure Things</strong><br>
        <em>Save budget on these customers</em><br><br>
        These users would convert anyway, with or without the ad. Showing them ads does not 
        hurt, but it also does not help. The ad spend is unnecessary - they were already 
        going to buy.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="segment-card sleeping-dogs">
        <strong>Sleeping Dogs</strong><br>
        <em>Avoid these customers</em><br><br>
        These users actually convert less when shown ads. This might seem counterintuitive, 
        but it happens - perhaps the ad feels intrusive, or it triggers negative associations. 
        Targeting these users actively hurts conversion rates.
        </div>
        """, unsafe_allow_html=True)
    
    # Segment distribution
    st.markdown('<p class="section-header">Segment Distribution in Our Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=segments['Segment'],
            values=segments['Count'],
            marker_colors=[SEGMENT_COLORS[s] for s in segments['Segment']],
            textinfo='label+percent',
            textposition='outside',
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title="Customer Distribution by Segment",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Uplift by segment
        segment_order = ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']
        segments_ordered = segments.set_index('Segment').loc[segment_order].reset_index()
        
        fig_uplift = go.Figure()
        
        fig_uplift.add_trace(go.Bar(
            x=segments_ordered['Segment'],
            y=segments_ordered['Mean_Uplift'],
            marker_color=[SEGMENT_COLORS[s] for s in segments_ordered['Segment']],
            text=segments_ordered['Mean_Uplift'].apply(lambda x: f'{x:.3f}%'),
            textposition='outside'
        ))
        
        fig_uplift.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig_uplift.update_layout(
            title="Average Predicted Uplift by Segment",
            xaxis_title="Segment",
            yaxis_title="Mean Uplift (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_uplift, use_container_width=True)
    
    # Key numbers
    st.markdown('<p class="section-header">Segment Statistics</p>', unsafe_allow_html=True)
    
    # Format the dataframe for display
    display_df = segments.copy()
    display_df['Count'] = display_df['Count'].apply(lambda x: f'{x:,}')
    display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f'{x:.1f}%')
    display_df['Mean_Uplift'] = display_df['Mean_Uplift'].apply(lambda x: f'{x:.4f}%')
    display_df.columns = ['Segment', 'Count', 'Percentage', 'Mean Uplift']
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>What the numbers tell us:</strong> The customer base splits roughly into quarters 
    across the four segments. About 25% are Persuadables with meaningful positive uplift, 
    and a similar proportion are Sleeping Dogs with negative uplift. The math here is 
    straightforward - if we can shift ad spend from Sleeping Dogs to Persuadables, we 
    should see significant improvement in campaign efficiency.
    </div>
    """, unsafe_allow_html=True)
    
    # Uplift distribution by segment
    st.markdown('<p class="section-header">Uplift Distribution Within Segments</p>', unsafe_allow_html=True)
    
    # Sample for visualization (full dataset is too large for histogram)
    sample_size = min(10000, len(predictions))
    predictions_sample = predictions.sample(n=sample_size, random_state=42)
    
    fig_dist = go.Figure()
    
    for segment in segment_order:
        seg_data = predictions_sample[predictions_sample['segment'] == segment]['uplift_pred'] * 100
        fig_dist.add_trace(go.Histogram(
            x=seg_data,
            name=segment,
            opacity=0.7,
            marker_color=SEGMENT_COLORS[segment],
            nbinsx=50
        ))
    
    fig_dist.update_layout(
        title="Distribution of Predicted Uplift by Segment",
        xaxis_title="Predicted Uplift (%)",
        yaxis_title="Count",
        barmode='overlay',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>Reading this chart:</strong> The Persuadables (green) cluster on the right with 
    positive uplift values, while Sleeping Dogs (red) cluster on the left with negative 
    values. Sure Things and Lost Causes are in the middle - they have near-zero uplift 
    because their behavior does not change much either way. The separation between segments 
    is reasonably clean, which suggests the model is capturing real differences in user behavior.
    </div>
    """, unsafe_allow_html=True)
    
    # Business recommendations
    st.markdown('<p class="section-header">Business Recommendations</p>', unsafe_allow_html=True)
    
    recommendations = pd.DataFrame({
        'Segment': ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs'],
        'Action': ['Target', 'Save Budget', 'Skip', 'Avoid'],
        'Rationale': [
            'Highest ROI - ads drive incremental conversions',
            'Would convert anyway - ad spend is unnecessary',
            'Will not convert regardless - no point in targeting',
            'Ads hurt conversion - actively counterproductive'
        ],
        'Budget Priority': ['High', 'Low', 'None', 'Exclude']
    })
    
    st.dataframe(recommendations, hide_index=True, use_container_width=True)
    
    # My take
    st.markdown('<p class="section-header">My Take on This</p>', unsafe_allow_html=True)
    
    st.markdown("""
    The segmentation results make intuitive sense, but there are a few things I would 
    flag for discussion:
    
    **On Persuadables:** These are the clear winners, but 25% of the customer base is 
    actually a pretty high proportion. It suggests there is significant room for targeting 
    optimization. If we are currently showing ads to everyone, we are wasting roughly 
    75% of our ad spend on users who either do not need it or are hurt by it.
    
    **On Sleeping Dogs:** The existence of this segment is the most interesting finding 
    to me. It is counterintuitive that ads would reduce conversion, but it is a real 
    phenomenon. Possible explanations include ad fatigue, perceived intrusiveness, or 
    triggering comparison shopping. Whatever the cause, the prescription is clear: 
    exclude these users from targeting.
    
    **On Sure Things:** This is the "hidden savings" segment. These users would buy anyway, 
    so ad spend here is pure waste. In a budget-constrained environment, reallocating 
    this spend to Persuadables is low-hanging fruit.
    
    **On Lost Causes:** No surprise here. Some users just are not in the market. The 
    model correctly identifies them and we should not waste resources.
    
    **The bottom line:** If I were making a recommendation to the marketing team, it 
    would be to implement segment-based targeting. Prioritize Persuadables, exclude 
    Sleeping Dogs, and de-prioritize Sure Things and Lost Causes. The expected efficiency 
    gain is substantial.
    """)


if __name__ == "__main__":
    main()


