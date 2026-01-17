"""
Uplift Modeling Dashboard v2 - Clean, Visual-First Design
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Uplift Modeling - Criteo Dataset",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - use resolve() to get absolute paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "visualizations" / "data"
SAMPLE_DATA = PROJECT_ROOT / "data" / "processed" / "criteo_uplift_sample.csv"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "criteo_uplift.parquet"

# Plotly dark template
PLOTLY_TEMPLATE = "plotly_dark"



@st.cache_data
def load_eda_summary():
    return pd.read_csv(DATA_DIR / "nb01_eda_summary.csv")

@st.cache_data
def load_outcome_distribution():
    return pd.read_csv(DATA_DIR / "nb01_outcome_distribution.csv")

@st.cache_data
def load_sample_data():
    return pd.read_csv(SAMPLE_DATA)

@st.cache_data
def load_full_data_stats():
    """Load full parquet and compute exposure->visit stats"""
    try:
        df = pd.read_parquet(RAW_DATA)
        # Users with exposure
        exposed_users = df[df['exposure'] == 1]
        exposed_count = len(exposed_users)
        exposed_and_visited = (exposed_users['visit'] == 1).sum()
        exposed_and_converted = (exposed_users['conversion'] == 1).sum()
        
        return {
            'total': len(df),
            'exposed_count': exposed_count,
            'exposed_and_visited': exposed_and_visited,
            'exposed_and_converted': exposed_and_converted,
            'visit_rate_given_exposure': exposed_and_visited / exposed_count * 100 if exposed_count > 0 else 0,
            'conversion_rate_given_exposure': exposed_and_converted / exposed_count * 100 if exposed_count > 0 else 0
        }
    except Exception as e:
        return None




def render_background_tab():
    """Tab 1: What is Uplift Modeling?"""
    
    st.header("What is Uplift Modeling?")
    st.write("")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**The Problem**")
        st.markdown("""
        You run an ad campaign. Some people convert. But here's the question:
        
        **Did they convert *because* of the ad, or would they have converted anyway?**
        """)
    
    with col_right:
        st.markdown("**The Solution**")
        st.markdown("""
        Traditional A/B testing tells you the *average* effect. 
        
        Uplift modeling tells you **who** is most affected by the treatment.
        """)
    
    st.write("")
    
    # --- Four Types of Users (2x2 grid) ---
    st.subheader("The Four Types of Users")
    st.write("")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 20px; border-radius: 10px; text-align: center; height: 140px; margin-bottom: 15px;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 8px;">Persuadables</div>
            <div style="font-size: 0.9rem; color: #d1fae5; margin-bottom: 8px;">Without ad: No | With ad: Yes</div>
            <div style="font-size: 0.85rem; color: #6ee7b7;">Target these users</div>
        </div>
        """, unsafe_allow_html=True)
    
    with row1_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%); padding: 20px; border-radius: 10px; text-align: center; height: 140px; margin-bottom: 15px;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 8px;">Sure Things</div>
            <div style="font-size: 0.9rem; color: #bfdbfe; margin-bottom: 8px;">Without ad: Yes | With ad: Yes</div>
            <div style="font-size: 0.85rem; color: #93c5fd;">Save your budget</div>
        </div>
        """, unsafe_allow_html=True)
    
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #374151 0%, #4b5563 100%); padding: 20px; border-radius: 10px; text-align: center; height: 140px;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 8px;">Lost Causes</div>
            <div style="font-size: 0.9rem; color: #d1d5db; margin-bottom: 8px;">Without ad: No | With ad: No</div>
            <div style="font-size: 0.85rem; color: #9ca3af;">Don't waste resources</div>
        </div>
        """, unsafe_allow_html=True)
    
    with row2_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%); padding: 20px; border-radius: 10px; text-align: center; height: 140px;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #fff; margin-bottom: 8px;">Sleeping Dogs</div>
            <div style="font-size: 0.9rem; color: #fecaca; margin-bottom: 8px;">Without ad: Yes | With ad: No</div>
            <div style="font-size: 0.85rem; color: #fca5a5;">Avoid at all costs</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.info("**In my mind:** The goal is simple - find the Persuadables and target them. Ignore the rest. That's where uplift modeling beats traditional targeting.")
    
    st.write("")
    
    # --- How to estimate uplift ---
    st.subheader("How Do We Estimate Uplift?")
    st.write("")
    
    st.markdown("""
    We can't observe the same person with AND without treatment. So we use **meta-learners** — 
    ML models that estimate individual treatment effects:
    """)
    
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **S-Learner** (Single)
        
        Train ONE model with treatment as a feature.
        
        `uplift = model(X, T=1) - model(X, T=0)`
        
        *Simple but can miss treatment effects*
        """)
    
    with col2:
        st.markdown("""
        **T-Learner** (Two)
        
        Train TWO separate models — one for treated, one for control.
        
        `uplift = model_t(X) - model_c(X)`
        
        *Works well with enough data*
        """)
    
    with col3:
        st.markdown("""
        **X-Learner** (Cross)
        
        Four-stage approach with counterfactual imputation.
        
        *Best for imbalanced treatment/control splits*
        """)
    
    st.write("")
    st.info("**In my mind:** T-Learner often wins when you have enough samples in both groups. X-Learner shines when control is tiny. S-Learner is a good baseline but tends to underestimate effects.")


def render_eda_tab():
    """Tab 2: Exploratory Data Analysis"""
    
    # Load data
    eda_summary = load_eda_summary()
    outcome_dist = load_outcome_distribution()
    sample_df = load_sample_data()
    
    # Helper to get metric
    def get_metric(name):
        row = eda_summary[eda_summary['Metric'] == name]
        return row['Value'].values[0] if len(row) > 0 else None
    
    total_samples = get_metric('Total Samples')
    n_features = get_metric('Number of Features')
    conversion_rate = get_metric('Conversion Rate %')
    ate = get_metric('ATE (percentage points)')
    
    # --- Header ---
    st.header("Exploratory Data Analysis")
    
    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", f"{total_samples/1e6:.1f}M")
    with col2:
        st.metric("Features", f"{int(n_features)}")
    with col3:
        st.metric("Conv. Rate", f"{conversion_rate:.2f}%")
    with col4:
        st.metric("ATE", f"+{ate:.3f}pp")
    
    # --- Sample Data ---
    st.subheader("Sample Data")
    st.dataframe(sample_df.head(10), use_container_width=True)
    
    # --- Missing Values ---
    st.subheader("Missing Values")
    
    missing = sample_df.isnull().sum()
    
    if missing.sum() == 0:
        st.success("No missing values in the dataset")
    else:
        st.dataframe(missing[missing > 0])
    
    # --- Treatment Split ---
    st.subheader("Treatment / Control Split")
    
    control_pct = get_metric('Control %')
    treatment_pct = get_metric('Treatment %')
    control_n = get_metric('Control Samples')
    treatment_n = get_metric('Treatment Samples')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_split = go.Figure(data=[go.Pie(
            labels=['Control', 'Treatment'],
            values=[control_n, treatment_n],
            hole=0.5,
            marker_colors=['#718096', '#4a9eff'],
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        
        fig_split.update_layout(
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            annotations=[dict(text=f'{total_samples/1e6:.1f}M', x=0.5, y=0.5, 
                             font_size=20, showarrow=False, font_color='white')]
        )
        
        st.plotly_chart(fig_split, use_container_width=True)
    
    with col2:
        st.info("""
**In my mind:** Why 85% treatment and only 15% control? This seems backwards from typical A/B tests.

In a product A/B test, you keep treatment small because it's risky. But this is an **ad campaign** - showing ads = making money. Every user in control is an ad you didn't show = lost revenue.

The 15% holdout is just enough to measure if the ads actually work.
""")
    
    # --- Target Variable Analysis ---
    st.subheader("Target Variables: What do they mean?")
    
    exposure_rate = get_metric('Exposure Rate %')
    visit_rate = get_metric('Visit Rate %')
    
    # Definitions table
    definitions = pd.DataFrame({
        'Metric': ['exposure', 'visit', 'conversion'],
        'Rate': [f'{exposure_rate:.2f}%', f'{visit_rate:.2f}%', f'{conversion_rate:.2f}%'],
        'Meaning': [
            'Did the user actually SEE the ad? (rendered, viewable)',
            'Did the user VISIT the advertiser\'s website? (from any source)',
            'Did the user CONVERT? (purchase, sign up, etc.)'
        ],
        'Note': [
            'Only ~3% of users actually saw the ad (blocking, below fold, etc.)',
            'Higher than exposure! Includes organic, direct, other sources',
            'The final goal - extremely rare in digital advertising'
        ]
    })
    st.dataframe(definitions, hide_index=True, use_container_width=True)
    
    # --- The Question: Why is Visit > Exposure? ---
    st.subheader("Wait... why is Visit rate > Exposure rate?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Show the rates
        fig_compare = go.Figure()
        
        fig_compare.add_trace(go.Bar(
            x=['Exposure', 'Visit', 'Conversion'],
            y=[exposure_rate, visit_rate, conversion_rate],
            marker_color=['#4a9eff', '#4ade80', '#f59e0b'],
            text=[f'{exposure_rate:.2f}%', f'{visit_rate:.2f}%', f'{conversion_rate:.2f}%'],
            textposition='outside',
            textfont=dict(size=14)
        ))
        
        fig_compare.update_layout(
            template=PLOTLY_TEMPLATE,
            height=350,
            xaxis_title='',
            yaxis_title='Rate (%)',
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40),
            title=dict(text='Overall Rates (All Users)', font=dict(size=14))
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="my-thought">
        <strong>In my mind:</strong> This confused me at first. But they're NOT a strict funnel.
        <br><br>
        <strong>Exposure = 3.06%</strong> → Only 3% of users actually <em>saw</em> the ad 
        (ad blocking, below fold, didn't load)
        <br><br>
        <strong>Visit = 4.70%</strong> → 4.7% visited the site, but this includes organic search, 
        direct traffic, other campaigns, etc.
        <br><br>
        A user can visit the website <strong>without</strong> seeing this specific ad. 
        Think of exposure as "did our billboard reach them" and visit as "did they show up at our store" - 
        people can show up without seeing the billboard.
        </div>
        """, unsafe_allow_html=True)
    
    # --- The REAL Funnel: Users who saw the ad ---
    st.subheader("The Closed Funnel: Among users who SAW the ad")
    
    # Try to load the full stats
    full_stats = load_full_data_stats()
    
    if full_stats:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Funnel for exposed users only
            fig_real_funnel = go.Figure(go.Funnel(
                y=['Saw Ad (Exposure)', 'Then Visited', 'Then Converted'],
                x=[100, full_stats['visit_rate_given_exposure'], full_stats['conversion_rate_given_exposure']],
                textinfo='value+percent initial',
                texttemplate='%{value:.1f}%',
                marker=dict(color=['#4a9eff', '#4ade80', '#f59e0b']),
                connector=dict(line=dict(color='#374151', width=2))
            ))
            
            fig_real_funnel.update_layout(
                template=PLOTLY_TEMPLATE,
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(text='Funnel: Users Who Saw The Ad', font=dict(size=14))
            )
            
            st.plotly_chart(fig_real_funnel, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            **Among the {full_stats['exposed_count']:,} users who actually saw the ad:**
            
            | Step | Count | Rate |
            |------|-------|------|
            | Saw Ad | {full_stats['exposed_count']:,} | 100% |
            | Then Visited | {full_stats['exposed_and_visited']:,} | {full_stats['visit_rate_given_exposure']:.1f}% |
            | Then Converted | {full_stats['exposed_and_converted']:,} | {full_stats['conversion_rate_given_exposure']:.2f}% |
            """)
            
            st.markdown("""
            <div class="my-thought">
            <strong>In my mind:</strong> NOW this looks like a proper funnel. Among users who 
            actually saw the ad, the visit and conversion rates make more sense as a sequential flow.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Loading full dataset to calculate exposure-based funnel...")
    
    # --- Treatment Effect ---
    st.subheader("Treatment Effect")
    
    relative_lift = get_metric('Relative Lift %')
    
    # Approximate control and treatment conversion rates
    control_conv = conversion_rate - (treatment_pct/100) * ate
    treatment_conv = conversion_rate + (control_pct/100) * ate
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_ate = go.Figure()
        
        fig_ate.add_trace(go.Bar(
            x=['Control', 'Treatment'],
            y=[control_conv, treatment_conv],
            marker_color=['#718096', '#4a9eff'],
            text=[f'{control_conv:.3f}%', f'{treatment_conv:.3f}%'],
            textposition='outside',
            textfont=dict(size=16, color='white')
        ))
        
        fig_ate.update_layout(
            template=PLOTLY_TEMPLATE,
            height=500,
            xaxis_title='',
            yaxis_title='Conversion Rate (%)',
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        
        st.plotly_chart(fig_ate, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Absolute Effect</div>
            <div class="metric-value">+{ate:.3f}pp</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Relative Lift</div>
            <div class="metric-value">+{relative_lift:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["Background", "EDA"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("""
        **Data Source:**
        - [Criteo Uplift Dataset](https://huggingface.co/datasets/criteo/criteo-uplift)
        - ~14M samples from RCT
        """)
        
        st.divider()
        
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.8rem;">
            Built by <a href="https://www.linkedin.com/in/sivagugan-jayachandran/" target="_blank" style="color: #4a9eff; text-decoration: none;">Sivagugan Jayachandran</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Render selected page
    if page == "Background":
        render_background_tab()
    elif page == "EDA":
        render_eda_tab()


if __name__ == "__main__":
    main()
