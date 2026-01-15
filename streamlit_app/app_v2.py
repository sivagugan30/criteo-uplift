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

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "visualizations" / "data"
SAMPLE_DATA = PROJECT_ROOT / "data" / "processed" / "criteo_uplift_sample.csv"
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "criteo_uplift.parquet"

# Plotly dark template
PLOTLY_TEMPLATE = "plotly_dark"

# Custom CSS - clean, minimal
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .my-thought {
        background-color: #1e2530;
        border-left: 3px solid #4a9eff;
        padding: 12px 16px;
        margin: 16px 0;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        color: #a0aec0;
    }
    .metric-box {
        background-color: #1e2530;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
    }
    h1, h2, h3 {
        color: #fff;
    }
    .source-link {
        background-color: #1e2530;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .source-link a {
        color: #4a9eff;
        text-decoration: none;
    }
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)


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


def render_header():
    """Render the header with source info"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Criteo Uplift Modeling")
        st.caption("Causal inference on a large-scale advertising RCT")
    
    with col2:
        st.markdown("""
        <div class="source-link">
            <strong>Dataset:</strong> Criteo AI Lab<br>
            <a href="https://huggingface.co/datasets/criteo/criteo-uplift" target="_blank">
                huggingface.co/datasets/criteo/criteo-uplift
            </a><br>
            <span style="color: #718096; font-size: 0.85rem;">~14M samples from randomized controlled trial</span>
        </div>
        """, unsafe_allow_html=True)


def render_background_tab():
    """Tab 1: What is Uplift Modeling?"""
    
    st.header("What is Uplift Modeling?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("The Problem")
        st.markdown("""
        You run an ad campaign. Some people convert. But here's the question:
        
        **Did they convert *because* of the ad, or would they have converted anyway?**
        
        Traditional A/B testing tells you the *average* effect. Uplift modeling tells you 
        **who** is most affected by the treatment.
        """)
        
        # Simple visual: 4 user types
        st.subheader("Four Types of Users")
        
        user_types = pd.DataFrame({
            'Type': ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs'],
            'Without Ad': ['No', 'Yes', 'No', 'Yes'],
            'With Ad': ['Yes', 'Yes', 'No', 'No'],
            'Action': ['Target', 'Save $', 'Skip', 'Avoid']
        })
        
        st.dataframe(user_types, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Causal Inference")
        st.markdown("""
        Causal inference answers: **"What would have happened if...?"**
        
        - What if this user *didn't* see the ad?
        - What if we *only* targeted high-value users?
        
        We can't observe both outcomes for the same person (the *fundamental problem of causal inference*), 
        so we use statistical methods to estimate it.
        """)
        
        st.markdown("""
        <div class="my-thought">
        <strong>In my mind:</strong> Think of it like this - you want to find people whose 
        behavior actually changes because of your action. Not people who were going to buy 
        anyway, and definitely not people who get annoyed by your ads.
        </div>
        """, unsafe_allow_html=True)
        
        # Simple causal diagram
        st.subheader("The Causal Question")
        
        fig = go.Figure()
        
        # Nodes
        fig.add_trace(go.Scatter(
            x=[0, 1, 2],
            y=[1, 0, 1],
            mode='markers+text',
            marker=dict(size=50, color=['#4a9eff', '#ff6b6b', '#4ade80']),
            text=['Treatment<br>(Ad)', 'Features<br>(X)', 'Outcome<br>(Convert)'],
            textposition='bottom center',
            textfont=dict(size=12, color='white'),
            hoverinfo='none'
        ))
        
        # Arrows
        fig.add_annotation(x=2, y=1, ax=0, ay=1, xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#4a9eff')
        fig.add_annotation(x=0, y=1, ax=1, ay=0, xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#ff6b6b')
        fig.add_annotation(x=2, y=1, ax=1, ay=0, xref='x', yref='y', axref='x', ayref='y',
                          showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#ff6b6b')
        
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            xaxis=dict(visible=False, range=[-0.5, 2.5]),
            yaxis=dict(visible=False, range=[-0.5, 1.5]),
            height=250,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_eda_tab():
    """Tab 2: Exploratory Data Analysis"""
    
    st.header("Exploratory Data Analysis")
    
    # Load data
    eda_summary = load_eda_summary()
    outcome_dist = load_outcome_distribution()
    sample_df = load_sample_data()
    
    # Helper to get metric
    def get_metric(name):
        row = eda_summary[eda_summary['Metric'] == name]
        return row['Value'].values[0] if len(row) > 0 else None
    
    # --- Dataset Overview ---
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_samples = get_metric('Total Samples')
    n_features = get_metric('Number of Features')
    conversion_rate = get_metric('Conversion Rate %')
    ate = get_metric('ATE (percentage points)')
    
    with col1:
        st.metric("Total Samples", f"{total_samples:,.0f}")
    with col2:
        st.metric("Features", f"{int(n_features)} (anonymized)")
    with col3:
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    with col4:
        st.metric("Treatment Effect", f"+{ate:.3f}pp")
    
    # --- Sample Data ---
    st.subheader("Sample Data (head 10)")
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
        st.markdown("""
        <div class="my-thought">
        <strong>In my mind:</strong> Why 85% treatment and only 15% control? This seems 
        backwards from typical A/B tests.
        <br><br>
        In a product A/B test, you keep treatment small because it's risky. But this is 
        an <strong>ad campaign</strong> - showing ads = making money. Every user in control 
        is an ad you didn't show = lost revenue.
        <br><br>
        The 15% holdout is just enough to measure if the ads actually work.
        </div>
        """, unsafe_allow_html=True)
    
    # --- Target Variable Analysis ---
    st.subheader("Target Variables: What do they mean?")
    
    # Definitions table
    definitions = pd.DataFrame({
        'Metric': ['exposure', 'visit', 'conversion'],
        'Meaning': [
            'Did the user actually SEE the ad? (viewability - rendered, above fold, not blocked)',
            'Did the user VISIT the advertiser\'s website? (from any source)',
            'Did the user CONVERT? (purchase, sign up, etc.)'
        ]
    })
    st.dataframe(definitions, hide_index=True, use_container_width=True)
    
    # --- The Question: Why is Visit > Exposure? ---
    st.subheader("Wait... why is Visit rate > Exposure rate?")
    
    exposure_rate = get_metric('Exposure Rate %')
    visit_rate = get_metric('Visit Rate %')
    
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
    st.subheader("The Real Funnel: Among users who SAW the ad")
    
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
            height=350,
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
        **Pages:**
        
        1. **Background** - What is uplift modeling and causal inference
        
        2. **EDA** - Exploratory data analysis of the Criteo dataset
        """)
        
        st.divider()
        
        st.caption("Built with Streamlit + Plotly")
    
    # Header
    render_header()
    
    st.divider()
    
    # Render selected page
    if page == "Background":
        render_background_tab()
    elif page == "EDA":
        render_eda_tab()


if __name__ == "__main__":
    main()
