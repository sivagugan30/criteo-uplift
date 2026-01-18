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
    page_title="Uplift Modeling - Criteo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - go up from streamlit_app to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "visualizations" / "data"
# Sample data now comes from predictions CSV (data/processed is gitignored)

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
    """Load sample data from predictions CSV (has features + outcomes)"""
    predictions = pd.read_csv(DATA_DIR / "nb02_predictions.csv")
    # Rename columns to match expected format
    sample = predictions[['y_true', 'treatment'] + [f'f{i}' for i in range(12)]].copy()
    sample = sample.rename(columns={'y_true': 'conversion'})
    return sample.head(1000)  # Just need a sample for display

@st.cache_data
def load_full_data_stats():
    """Return pre-computed stats (raw parquet not available on Streamlit Cloud)"""
    try:
        # These stats are from the full 25M dataset analysis
        return {
            'total': 25309483,
            'exposed_count': 21551779,
            'exposed_and_visited': 816513,
            'exposed_and_converted': 11853,
            'visit_rate_given_exposure': 3.79,
            'conversion_rate_given_exposure': 0.055
        }
    except Exception as e:
        return None


def render_background_tab():
    """Tab 1: What is Uplift Modeling?"""
    
    st.header("What is Uplift Modeling?")
    st.write("")
    
    # Problem vs Solution - side by side
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("The Problem")
        st.markdown("""
        You run an ad campaign. Some people convert. But here's the question:
        
        **Did they convert *because* of the ad, or would they have converted anyway?**
        """)
    
    with col_right:
        st.subheader("The Solution")
        st.markdown("""
        Traditional A/B testing tells you the *average* effect. 
        
        Uplift modeling tells you **who** is most affected by the treatment.
        """)
    
    st.write("")
    
    # Four Types of Users - 2x2 grid with styled cards
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
            <div style="font-size: 0.85rem; color: #93c5fd;">Save our budget</div>
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
    st.divider()
    st.caption("📄 **The OG Paper:** This project uses the Criteo Uplift Dataset, released by Criteo Research in 2018. It's the largest public uplift modeling benchmark (25M samples from a randomized control trial). [Read the paper](https://bitlater.github.io/files/large-scale-benchmark_comAH.pdf)")


def render_data_story_tab():
    """Tab 2: The Data Story"""
    
    st.header("The Data Story")
    
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
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{total_samples/1e6:.1f}M")
    with col2:
        st.metric("Features", f"{int(n_features)}")
        st.caption("(anonymized)")
    with col3:
        st.metric("Conv. Rate", f"{conversion_rate:.2f}%")
    with col4:
        st.metric("Avg. Treatment Effect", f"+{ate:.3f}%")
    
    st.write("")
    
    # IDA - Initial Data Assessment (collapsed by default)
    with st.expander("IDA (Initial Data Assessment)", expanded=False):
        # Sample Data
        st.subheader("Sample Data")
        st.dataframe(sample_df.head(10), use_container_width=True)
        
        # Missing Values
        st.subheader("Missing Values")
        
        missing = sample_df.isnull().sum()
        
        if missing.sum() == 0:
            st.success("No missing values in the dataset")
        else:
            st.dataframe(missing[missing > 0])
    
    # Treatment Split
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
    
    # Target Variable Analysis
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
        ]
    })
    st.dataframe(definitions, hide_index=True, use_container_width=True)
    
    # The Question: Why is Visit > Exposure?
    st.subheader("Wait... why is Visit rate > Exposure rate?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
        st.info("""
**In my mind:** This confused me at first. But they're NOT a strict funnel.

**Exposure = 3.06%** → Only 3% of users actually *saw* the ad (ad blocking, below fold, didn't load)

**Visit = 4.70%** → 4.7% visited the site, but this includes organic search, direct traffic, other campaigns, etc.

A user can visit the website **without** seeing this specific ad. Think of exposure as "did our billboard reach them" and visit as "did they show up at our store" - people can show up without seeing the billboard.
""")
    
    # The REAL Funnel: Users who saw the ad
    st.subheader("The Closed Funnel: Among users who SAW the ad")
    
    full_stats = load_full_data_stats()
    
    if full_stats:
        col1, col2 = st.columns([1, 1])
        
        with col1:
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
            
            st.info("**In my mind:** NOW this looks like a proper funnel. Among users who actually saw the ad, the visit and conversion rates make more sense as a sequential flow.")
    else:
        st.info("Loading full dataset to calculate exposure-based funnel...")
    
    # Treatment Effect
    st.subheader("Treatment Effect")
    
    relative_lift = get_metric('Relative Lift %')
    
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
            height=400,
            xaxis_title='',
            yaxis_title='Conversion Rate (%)',
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        
        st.plotly_chart(fig_ate, use_container_width=True)
    
    with col2:
        st.metric("Absolute Effect", f"+{ate:.3f}pp")
        st.metric("Relative Lift", f"+{relative_lift:.0f}%")
    
    st.write("")
    st.info("""
**In my mind:** This is classic A/B testing - I got the **ATE** (Average Treatment Effect). The ad works, +59% lift, ship it, right?

But wait. This is the *average* across 14 million users. Some users might have +200% lift (Persuadables), others might be -50% (Sleeping Dogs). The average hides all of this.

What I really need is the **CATE** - Conditional Average Treatment Effect. *How does the treatment effect vary by user?* That's where uplift modeling comes in. Instead of asking "does the ad work?", I ask "for whom does it work best?"
""")


def render_cate_tab():
    """Tab 3: From ATE to CATE"""
    
    st.header("From ATE to CATE")
    st.write("")
    
    st.markdown("""
    In the previous section, I showed that the **ATE (Average Treatment Effect)** tells us the ad works on average.
    But I want to know **who** it works for - that's the **CATE (Conditional Average Treatment Effect)**.
    """)
    
    st.write("")
    
    # The fundamental problem
    st.subheader("The Fundamental Problem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        I can't observe the same person **with** AND **without** treatment.
        
        - User A saw the ad → converted ✓
        - What if User A *didn't* see the ad? 🤷
        
        This is the **fundamental problem of causal inference**.
        """)
    
    with col2:
        st.markdown("""
        **The solution:** Use **meta-learners** - ML models that estimate 
        individual treatment effects by learning patterns from the 
        treated and control groups.
        """)
    
    st.write("")
    st.divider()
    st.write("")
    
    # Meta-learners
    st.subheader("Meta-Learners: How I Estimate CATE")
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
        
        Train TWO separate models - one for treated, one for control.
        
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
    st.info("**In my mind:** T-Learner often wins when you have enough samples in both groups. X-Learner shines when control is tiny (like our 15%!). S-Learner is a good baseline but tends to underestimate effects.")
    
    st.write("")
    
    # Why this matters
    st.subheader("Why This Matters")
    st.write("")
    
    st.markdown("""
    Once I have CATE estimates for each user, I can:
    
    1. **Rank users** by predicted uplift
    2. **Identify Persuadables** (high CATE) vs Sleeping Dogs (negative CATE)
    3. **Optimize targeting** - show ads only to users who will actually respond
    4. **Calculate ROI** - estimate incremental revenue from smarter targeting
    """)
    
    st.write("")
    st.info("**In my mind:** This is where uplift modeling becomes powerful. Instead of showing ads to everyone (expensive) or no one (no revenue), I find the sweet spot - the users whose behavior actually changes because of the ad.")
    
    st.write("")
    with st.expander("Uber's Take on Uplift Modeling", expanded=False):
        st.markdown("""
        I found this talk from Uber's data science team super insightful. I think this was around the launch of the CausalML library. They explain how they moved from propensity modeling to uplift modeling, and walk through the library basics.
        
        [▶️ Watch on YouTube: Uplift Modeling - From Causal Inference to Personalization](https://www.youtube.com/watch?v=2J9j7peWQgI)
        """)


@st.cache_data
def load_model_metrics():
    return pd.read_csv(DATA_DIR / "nb02_model_metrics.csv")

@st.cache_data
def load_qini_curves():
    return pd.read_csv(DATA_DIR / "nb02_qini_curves.csv")

@st.cache_data
def load_decile_analysis():
    return pd.read_csv(DATA_DIR / "nb04_decile_analysis.csv")

@st.cache_data
def load_customer_segments():
    return pd.read_csv(DATA_DIR / "nb05_customer_segments.csv")

@st.cache_data
def load_shap_importance():
    return pd.read_csv(DATA_DIR / "nb05_shap_importance.csv")

@st.cache_data
def load_predictions_with_segments():
    return pd.read_csv(DATA_DIR / "nb05_predictions_with_segments.csv")


def render_model_evaluation_tab():
    """Tab 4: Model Evaluation"""
    
    st.header("Model Evaluation")
    st.write("")
    
    st.markdown("""
    I trained S-Learner, T-Learner, and X-Learner on the Criteo dataset using the **causalml** library with **XGBoost** as the base classifier.
    """)
    
    with st.expander("Training Details", expanded=False):
        st.markdown("""
        **Stack:**
        - `causalml` for meta-learners (BaseSClassifier, BaseTClassifier, BaseXClassifier)
        - `XGBoost` as the base model (handles missing values, fast training)
        - 80/20 train-test split
        
        **Challenges I faced:**
        - Memory issues with 14M rows → had to sample for prototyping, then train on full data
        - X-Learner took 4x longer to train (four-stage approach)
        - Used default hyperparameters (n_estimators=100, max_depth=5) - tuning is a future improvement
        """)
    
    with st.expander("CATE Dependence: How Uplift Varies with Features", expanded=False):
        st.markdown("""
        **CATE Dependence Plots** show how the treatment effect (uplift) varies with feature values.
        
        - **Positive slope:** Higher feature values → more responsive to treatment
        - **Negative slope:** Higher feature values → less responsive
        - **Flat line:** Feature doesn't affect treatment response
        """)
        
        # Load predictions for CATE dependence
        predictions_dep = load_predictions_with_segments()
        feature_cols_dep = [f'f{i}' for i in range(12)]
        
        # Get top 4 important features
        shap_imp = load_shap_importance()
        top_features = shap_imp['Feature'].head(4).tolist()
        
        fig_cate = make_subplots(rows=2, cols=2, subplot_titles=top_features)
        
        for idx, feature in enumerate(top_features):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            feature_values = predictions_dep[feature].values
            uplift_values = predictions_dep['uplift_pred'].values * 100
            
            # Bin the data for trend line
            n_bins = 20
            bin_edges = np.percentile(feature_values, np.linspace(0, 100, n_bins + 1))
            bin_centers = []
            bin_means = []
            
            for i in range(n_bins):
                mask = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i+1])
                if mask.sum() > 0:
                    bin_centers.append(feature_values[mask].mean())
                    bin_means.append(uplift_values[mask].mean())
            
            # Scatter (sample for performance)
            sample_idx = np.random.choice(len(feature_values), size=min(2000, len(feature_values)), replace=False)
            fig_cate.add_trace(
                go.Scatter(
                    x=feature_values[sample_idx],
                    y=uplift_values[sample_idx],
                    mode='markers',
                    marker=dict(size=3, color='steelblue', opacity=0.3),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            # Trend line
            fig_cate.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=bin_means,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Trend',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Zero line
            fig_cate.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
        
        fig_cate.update_layout(
            template=PLOTLY_TEMPLATE,
            height=500,
            margin=dict(l=40, r=20, t=40, b=40)
        )
        
        fig_cate.update_xaxes(title_text="Feature Value")
        fig_cate.update_yaxes(title_text="Uplift (%)")
        
        st.plotly_chart(fig_cate, use_container_width=True)
        
        st.caption("*Red trend line shows average uplift at each feature value. Points are individual users.*")
    
    st.write("")
    st.markdown("Now, how do I know which one is actually good at predicting uplift?")
    
    # Load data
    model_metrics = load_model_metrics()
    qini_curves = load_qini_curves()
    decile_analysis = load_decile_analysis()
    
    colors = {'T-Learner': '#4ade80', 'S-Learner': '#4a9eff', 'X-Learner': '#f59e0b'}
    
    st.write("")
    
    # --- SECTION 1: Calibration Check (Predicted vs Actual by Decile) ---
    st.subheader("1. Calibration Check: Predicted vs Actual Uplift")
    st.write("")
    
    # Dropdown to select model
    selected_model = st.selectbox(
        "Select a model to inspect:",
        ['T-Learner', 'S-Learner', 'X-Learner']
    )
    
    st.write("")
    
    # Get data for selected model
    model_data = decile_analysis[decile_analysis['Model'] == selected_model]
    
    # Create predicted vs actual chart
    fig_calibration = go.Figure()
    
    fig_calibration.add_trace(go.Bar(
        x=[f"D{d}" for d in model_data['Decile']],
        y=model_data['Avg Predicted Uplift (%)'],
        name='Predicted Uplift',
        marker_color=colors[selected_model],
        opacity=0.6
    ))
    
    fig_calibration.add_trace(go.Bar(
        x=[f"D{d}" for d in model_data['Decile']],
        y=model_data['Actual Uplift (%)'],
        name='Actual Uplift',
        marker_color=colors[selected_model]
    ))
    
    fig_calibration.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_title='Decile (D1 = Top 10% predicted uplift)',
        yaxis_title='Uplift (%)',
        barmode='group',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=f'{selected_model}: Predicted vs Actual Uplift by Decile', font=dict(size=14))
    )
    
    st.plotly_chart(fig_calibration, use_container_width=True)
    
    st.info("""
**In my mind:** Okay, I'll be honest - I made a mistake here. I plotted this calibration chart thinking it would help me compare models. But then I read more about uplift evaluation and realized: this isn't a classification problem, it's a **ranking problem**.

I don't actually care if the model predicts 0.5% or 5% uplift - I care if it correctly **ranks** users from highest to lowest. A model could be terribly calibrated but still be the best ranker.

So let me try something I read about online: the Qini curve.
""")
    
    st.write("")
    st.divider()
    st.write("")
    
    # --- SECTION 2: Qini Curves ---
    st.subheader("2. Qini Curves: Comparing Ranking Performance")
    st.write("")
    
    with st.expander("What is a Qini Curve?", expanded=False):
        st.markdown("""
        The **Qini curve** measures how well a model ranks users by uplift:
        
        1. Rank all users by predicted uplift (highest first)
        2. Start targeting from the top
        3. Plot cumulative incremental conversions (Qini value = uplift × users targeted)
        
        **Good model:** Curve rises steeply at first (top users have high uplift), then flattens
        
        **Random line (spray-and-pray):** A straight diagonal from (0, 0) to the final Qini value. Why?
        - With random targeting, each user you pick has the same expected uplift (the average)
        - At 10% targeted, you capture ~10% of total lift. At 50%, you capture ~50%.
        - No prioritization = linear relationship = straight line
        
        **Key insight:** At 100%, ALL curves meet at the same point (total incremental conversions = ATE × N). The order doesn't matter when you target everyone.
        
        *The gap between your model and the random line = the value of having a model.*
        """)
    
    st.write("")
    
    fig_qini = go.Figure()
    
    for model in ['T-Learner', 'S-Learner', 'X-Learner']:
        model_qini = qini_curves[qini_curves['model'] == model]
        fig_qini.add_trace(go.Scatter(
            x=model_qini['percentile'] * 100,
            y=model_qini['qini'],
            mode='lines',
            name=model,
            line=dict(color=colors[model], width=2)
        ))
    
    # Add random line (diagonal from 0 to final Qini value)
    # At 100%, all models converge to same point: ATE × N (total incremental conversions)
    # Random targeting = no ranking benefit, so it's a straight line to that endpoint
    final_qini = qini_curves[qini_curves['percentile'] == 1.0]['qini'].iloc[0]
    fig_qini.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, final_qini],
        mode='lines',
        name='Random (spray-and-pray)',
        line=dict(color='#718096', width=2, dash='dash')
    ))
    
    fig_qini.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        xaxis_title='% of Population Targeted',
        yaxis_title='Qini (Cumulative Uplift)',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig_qini, use_container_width=True)
    
    st.info("""
**In my mind:** NOW I can compare models. Look at how **T-Learner (green)** rises fastest early on. If I can only target 20% of users, T-Learner picks the best ones.

Notice the **random line** (dashed gray). That's what spray-and-pray looks like: target 50% randomly, capture 50% of the lift. No acceleration.

All curves meet at 100% because when you target everyone, order doesn't matter. The total incremental conversions = ATE × N, regardless of the model.

The **gap between model and random** = the value of having a model. T-Learner's gap is largest, so it adds the most value.
""")
    
    st.write("")
    st.divider()
    st.write("")
    
    # --- SECTION 3: Qini Coefficient & AUUC ---
    st.subheader("3. Qini Coefficient & AUUC")
    st.write("")
    
    with st.expander("What's the difference between Qini Coefficient and AUUC?", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Qini Coefficient**
            
            - The **area between** the model's Qini curve and the random line
            - Measures how much **better** your model is vs. random targeting
            - More interpretable: "How many extra conversions do I get by using my model vs. spray-and-pray?"
            
            *T-Learner's 35.19 ≈ 35 extra conversions per 10K users vs random*
            """)
        
        with col2:
            st.markdown("""
            **AUUC (Area Under Uplift Curve)**
            
            - The **area under** the uplift curve (rate, not cumulative)
            - Also measures ranking quality, similar purpose
            - The number is abstract and relative
            
            *Good for: "Which model ranks users better overall?"*
            """)
    
    st.write("")
    
    # Metrics comparison
    st.markdown("**Qini Coefficient** (higher = more incremental conversions)")
    col1, col2, col3 = st.columns(3)
    
    for i, model in enumerate(['T-Learner', 'S-Learner', 'X-Learner']):
        row = model_metrics[model_metrics['Model'] == model].iloc[0]
        with [col1, col2, col3][i]:
            st.metric(model, f"{row['Qini_Coefficient']:.2f}")
    
    st.write("")
    
    st.markdown("**AUUC** (higher = better ranking across all percentiles)")
    col1, col2, col3 = st.columns(3)
    
    for i, model in enumerate(['T-Learner', 'S-Learner', 'X-Learner']):
        row = model_metrics[model_metrics['Model'] == model].iloc[0]
        with [col1, col2, col3][i]:
            st.metric(model, f"{row['AUUC']*100:.3f}%")
    
    st.write("")
    st.info("""
**In my mind:** Both metrics point to **T-Learner** as the winner. But I'll be honest - AUUC is a easy mathematical metric but it's hard to explain to business stakeholders.

The Qini Coefficient is more useful: T-Learner's 35.19 means if I target 10K users using T-Learner's ranking (top predicted uplift first), I'd get ~35 more conversions than if I just picked 10K users at random or blasted ads to everyone equally.
""")
    
    st.write("")
    st.divider()
    st.write("")
    
    # --- SECTION 4: Deploy vs Research ---
    st.subheader("4. What I Would Deploy vs. Research")
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
**Deploy: T-Learner**

Best Qini. Best AUUC.

Ship it.
""")
    
    with col2:
        st.warning("""
**Research: X-Learner**

X-Learner *should* win with an 85/15 split. It didn't.

I don't fully understand why yet.
""")
    
    st.write("")
    st.info("""
**In my mind:** The Qini curves look great offline. But what happens when I deploy?

- Does the uplift hold when user behavior shifts?
- What happens when I only target Persuadables and stop learning about the rest? (exploration vs. exploitation)

There's a lot to learn here. I've only scratched the surface.

The real question now - which users are Persuadables and which are Sleeping Dogs? Who should I actually spend ads on?
""")
    
    with st.expander("📝 A Note on Propensity Model Evaluation", expanded=False):
        st.markdown("""
        While working on this, I wondered: *"How would I evaluate a propensity model using deciles?"*
        
        My journey: Lift by decile has no single number. AUC-ROC isn't decile-based. KS Statistic measures separation at one threshold, not the full ranking.
        
        The answer: **Spearman correlation** between decile number and conversion rate. A good propensity model should have a negative slope (D1 highest, D10 lowest). Correlation close to -1.0 = great model. Simple, intuitive, decile-based.
        """)
    
    with st.expander("📄 Research Questions I'm Still Exploring", expanded=False):
        st.markdown("""
        - What do "non-CATE models" mean and how can they outperform proper CATE estimators?
        - Why are standard Qini curves high-variance on RCT data?
        - Why is the usual Qini "unnecessarily noisy" from random RCT sampling?
        - How do the RCT split and the ML split coexist?
        - What is a "separate prediction model for baseline outcome"?
        - How are adjusted outcomes computed and fed into Qini?
        - How do control units contribute to evaluation without individual counterfactuals?
        - Is adjusted Y only for test-set evaluation?
        - How do authors estimate variance reduction empirically?
        
        Sources: [Variance Reduction in Uplift (arXiv)](https://arxiv.org/pdf/2210.02152.pdf) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S037722172300721X)
        """)


def render_customer_segmentation_tab():
    """Tab 5: Customer Segmentation"""
    
    st.header("Customer Segmentation")
    st.write("")
    
    st.markdown("""
    Using T-Learner's predictions, I segmented users into the four uplift profiles using simple rules.
    """)
    
    # Load data
    shap_importance = load_shap_importance()
    predictions = load_predictions_with_segments()
    
    st.write("")
    
    # --- Segmentation Rules ---
    with st.expander("Segmentation Rules (Order Matters!)", expanded=True):
        st.markdown("""
        | Segment | Rule | Rationale |
        |---------|------|-----------|
        | **Sleeping Dogs** | `uplift < -0.5%` | Treatment hurts - check first for safety |
        | **Sure Things** | `baseline_prob > 50%` | Will convert anyway - save budget |
        | **Persuadables** | `uplift > 0.1%` | Positive incremental value from treatment |
        | **Lost Causes** | Everything else | Low or no uplift |
        """)
    
    st.write("")
    
    # Apply rule-based segmentation (order matters!)
    def apply_rules(row):
        uplift = row['uplift_pred']
        baseline = row['baseline_prob']
        
        # Rule 1: Sleeping Dogs - treatment hurts (check first for safety)
        if uplift < -0.005:  # -0.5%
            return 'Sleeping Dogs'
        # Rule 2: Sure Things - high baseline, will convert anyway
        elif baseline > 0.50:  # 50%
            return 'Sure Things'
        # Rule 3: Persuadables - positive uplift from treatment
        elif uplift > 0.001:  # 0.1%
            return 'Persuadables'
        # Rule 4: Lost Causes - everything else
        else:
            return 'Lost Causes'
    
    predictions['rule_segment'] = predictions.apply(apply_rules, axis=1)
    
    # Calculate segment stats
    segment_stats = predictions.groupby('rule_segment').agg({
        'uplift_pred': ['count', 'mean']
    }).reset_index()
    segment_stats.columns = ['Segment', 'Count', 'Mean_Uplift']
    segment_stats['Percentage'] = segment_stats['Count'] / len(predictions) * 100
    
    # Reorder segments
    segment_order = ['Persuadables', 'Sure Things', 'Lost Causes', 'Sleeping Dogs']
    segment_stats['Segment'] = pd.Categorical(segment_stats['Segment'], categories=segment_order, ordered=True)
    segment_stats = segment_stats.sort_values('Segment')
    
    # --- SECTION 1: Segment Distribution ---
    st.subheader("1. Segment Distribution")
    st.write("")
    
    # Segment colors
    segment_colors = {
        'Persuadables': '#4ade80',
        'Sure Things': '#4a9eff', 
        'Lost Causes': '#718096',
        'Sleeping Dogs': '#ef4444'
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_segments = go.Figure(data=[go.Pie(
            labels=segment_stats['Segment'],
            values=segment_stats['Count'],
            hole=0.5,
            marker_colors=[segment_colors[s] for s in segment_stats['Segment']],
            textinfo='label+percent',
            textfont=dict(size=12)
        )])
        
        fig_segments.update_layout(
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        # Create summary table
        rules = {
            'Persuadables': 'uplift > 0.1%',
            'Sure Things': 'baseline > 50%',
            'Lost Causes': 'everything else',
            'Sleeping Dogs': 'uplift < -0.5%'
        }
        
        table_df = segment_stats.copy()
        table_df['Rule'] = table_df['Segment'].map(rules)
        table_df['Count'] = table_df['Count'].apply(lambda x: f"{x:,}")
        table_df['Percentage'] = table_df['Percentage'].apply(lambda x: f"{x:.1f}%")
        table_df = table_df[['Segment', 'Rule', 'Count', 'Percentage']]
        
        st.dataframe(table_df, hide_index=True, use_container_width=True)
    
    st.write("")
    st.info("**In my mind:** This is a snapshot using fixed thresholds. Useful for understanding our user base, but not how I'd run a campaign.")
    
    st.write("")
    st.divider()
    st.write("")
    
    # --- SECTION 2: Explore Individual Users ---
    st.subheader("2. Explore a Random User")
    st.write("")
    
    st.markdown("Use the slider to pick a user and see why they're in their segment.")
    
    st.write("")
    
    # Slider to select user
    max_users = len(predictions) - 1
    user_idx = st.slider("Select User Index", 0, min(max_users, 1000), 421)
    
    user = predictions.iloc[user_idx]
    user_segment = user['rule_segment']
    user_uplift = user['uplift_pred']
    user_baseline = user['baseline_prob']
    median_baseline = predictions['baseline_prob'].median()
    
    st.write("")
    
    # Highlight: Customer Scorecard
    action_map = {
        'Persuadables': ('Target', '#4ade80', 'Treatment increases conversion'),
        'Sure Things': ('Save Budget', '#4a9eff', 'Converts anyway'),
        'Lost Causes': ('Skip', '#718096', 'Won\'t convert regardless'),
        'Sleeping Dogs': ('Avoid', '#ef4444', 'Treatment hurts conversion')
    }
    action, color, reason = action_map.get(user_segment, ('?', '#718096', ''))
    
    # Scorecard using Streamlit columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.caption("Segment")
        if user_segment == 'Persuadables':
            st.success(f"**{user_segment}**")
        elif user_segment == 'Sleeping Dogs':
            st.error(f"**{user_segment}**")
        elif user_segment == 'Sure Things':
            st.info(f"**{user_segment}**")
        else:
            st.warning(f"**{user_segment}**")
    
    with col2:
        st.metric("Predicted Uplift", f"{user_uplift*100:.3f}%")
    
    with col3:
        st.metric("Baseline Prob", f"{user_baseline*100:.3f}%")
    
    with col4:
        st.metric("Action", action)
    
    st.caption(f"*{reason}*")
    
    # Why this segment? (rule-based explanation)
    if user_segment == 'Sleeping Dogs':
        st.caption(f"**Why this segment?** Uplift ({user_uplift*100:.3f}%) < -0.5% threshold → Treatment hurts this user")
    elif user_segment == 'Sure Things':
        st.caption(f"**Why this segment?** Baseline ({user_baseline*100:.2f}%) > 50% threshold → Will convert without treatment")
    elif user_segment == 'Persuadables':
        st.caption(f"**Why this segment?** Uplift ({user_uplift*100:.3f}%) > 0.1% threshold → Treatment drives conversion")
    else:
        st.caption(f"**Why this segment?** Low uplift ({user_uplift*100:.3f}%) + low baseline ({user_baseline*100:.2f}%) → Unlikely to convert either way")
    
    st.write("")
    st.divider()
    
    # Comparative framing - what makes this user different?
    st.markdown("**Why This Predicted Uplift?**")
    st.caption(f"Feature contributions that explain the {user_uplift*100:.3f}% uplift prediction")
    
    # Get user's feature values
    feature_cols = [f'f{i}' for i in range(12)]
    user_features = user[feature_cols]
    
    # Calculate deviations from mean
    mean_values = predictions[feature_cols].mean()
    std_values = predictions[feature_cols].std()
    
    # Get total importance for normalization
    total_importance = shap_importance['Importance'].sum()
    
    contributions = []
    for f in feature_cols:
        user_val = user_features[f]
        mean_val = mean_values[f]
        std_val = std_values[f] if std_values[f] > 0.001 else 1
        importance = shap_importance[shap_importance['Feature'] == f]['Importance'].values[0]
        
        # How many std deviations from mean
        deviation = (user_val - mean_val) / std_val
        
        # Weighted contribution to uplift
        importance_pct = importance / total_importance
        contribution_pct = deviation * importance_pct * user_uplift * 100
        
        contributions.append({
            'Feature': f,
            'Value': user_val,
            'Mean': mean_val,
            'Deviation': deviation,
            'Contribution': contribution_pct
        })
    
    contrib_df = pd.DataFrame(contributions).sort_values('Contribution', key=abs, ascending=True)
    
    # Find standout features (most unusual)
    standout_high = contrib_df[contrib_df['Deviation'] > 0.5].nlargest(2, 'Deviation')
    standout_low = contrib_df[contrib_df['Deviation'] < -0.5].nsmallest(2, 'Deviation')
    
    # Horizontal bar chart - show deviation weighted by importance
    fig_contrib = go.Figure()
    
    colors = ['#ef4444' if c < 0 else '#4ade80' for c in contrib_df['Contribution']]
    
    # Create hover text with comparison info
    hover_texts = []
    for _, row in contrib_df.iterrows():
        direction = "above" if row['Deviation'] > 0 else "below"
        hover_texts.append(f"{row['Feature']}: {row['Value']:.2f}<br>Average: {row['Mean']:.2f}<br>{abs(row['Deviation']):.1f}σ {direction} average")
    
    fig_contrib.add_trace(go.Bar(
        y=contrib_df['Feature'],
        x=contrib_df['Contribution'],
        orientation='h',
        marker_color=colors,
        text=[f"{abs(d):.1f}σ {'↑' if d > 0 else '↓'}" for d in contrib_df['Deviation']],
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig_contrib.add_vline(x=0, line_color='white', line_width=1)
    
    fig_contrib.update_layout(
        template=PLOTLY_TEMPLATE,
        height=350,
        xaxis_title='← Lowers Uplift | Raises Uplift →',
        yaxis_title='',
        showlegend=False,
        margin=dict(l=40, r=60, t=20, b=40)
    )
    
    st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Find the single highest-impact feature (by absolute contribution)
    top_feature = contrib_df.loc[contrib_df['Contribution'].abs().idxmax()]
    
    # Get feature's global importance percentage
    feature_importance = shap_importance[shap_importance['Feature'] == top_feature['Feature']]['Importance'].values[0]
    importance_pct = (feature_importance / total_importance) * 100
    
    direction = "above" if top_feature['Deviation'] > 0 else "below"
    impact = "raises" if top_feature['Contribution'] > 0 else "lowers"
    
    # Just 1 line summary below the chart
    st.caption(f"Biggest driver: **{top_feature['Feature']}** ({abs(top_feature['Deviation']):.1f}σ {direction} avg) {impact} uplift by {abs(top_feature['Contribution']):.4f}%")
    
    with st.expander("See the math", expanded=False):
        st.markdown(f"**{top_feature['Feature']}** = {top_feature['Value']:.2f} (avg: {top_feature['Mean']:.2f})")
        st.caption(f"This feature is {abs(top_feature['Deviation']):.1f}σ {direction} average, and it's {importance_pct:.0f}% important globally.")
        
        st.code(f"""
Contribution = (deviation from avg) × (feature importance) × (predicted uplift)

{top_feature['Feature']} contribution:
= {top_feature['Deviation']:+.2f}σ × {importance_pct:.0f}% × {user_uplift*100:.3f}%
= {top_feature['Contribution']:+.4f}%
""", language=None)
        
        st.markdown("**User's Features:**")
        st.dataframe(user_features.to_frame().T, use_container_width=True)
    
    st.write("")
    st.divider()
    st.write("")
    
    # --- SECTION 3: Campaign Targeting Strategy ---
    st.subheader("3. Campaign Targeting Strategy")
    st.write("")
    
    st.markdown("""
    The rule-based segmentation above is useful for *understanding* the data. But for actually *running* a campaign?
    
    **Rank users by predicted uplift and target the top X% based on our budget.**
    """)
    
    st.write("")
    
    # Load Qini data for T-Learner
    qini_curves = load_qini_curves()
    t_learner_qini = qini_curves[qini_curves['model'] == 'T-Learner'].copy()
    
    # Slider for budget
    budget_pct = st.slider("Campaign Budget (% of users to target)", 5, 100, 50, 5)
    
    st.write("")
    
    # Get the Qini value at the selected percentage
    closest_row = t_learner_qini.iloc[(t_learner_qini['percentile'] * 100 - budget_pct).abs().argsort()[:1]]
    qini_at_budget = closest_row['qini'].values[0]
    max_qini = t_learner_qini['qini'].max()
    pct_of_max = (qini_at_budget / max_qini) * 100 if max_qini > 0 else 0
    
    # Create Qini chart with vertical line
    fig_targeting = go.Figure()
    
    # Qini curve
    fig_targeting.add_trace(go.Scatter(
        x=t_learner_qini['percentile'] * 100,
        y=t_learner_qini['qini'],
        mode='lines',
        name='T-Learner',
        line=dict(color='#4ade80', width=3)
    ))
    
    # Vertical line at budget
    fig_targeting.add_vline(
        x=budget_pct, 
        line_dash="dash", 
        line_color="#f59e0b",
        line_width=2
    )
    
    # Horizontal line showing captured lift
    fig_targeting.add_hline(
        y=qini_at_budget,
        line_dash="dot",
        line_color="#f59e0b",
        line_width=1
    )
    
    # Add annotation
    fig_targeting.add_annotation(
        x=budget_pct,
        y=qini_at_budget,
        text=f"Target {budget_pct}% → Capture {pct_of_max:.0f}% of lift",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f59e0b",
        font=dict(color="white", size=12),
        bgcolor="#1e2530",
        borderpad=4
    )
    
    fig_targeting.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        xaxis_title='% of Users Targeted (ranked by predicted uplift)',
        yaxis_title='Cumulative Uplift (Qini)',
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig_targeting, use_container_width=True)
    
    # Business metrics
    cost_savings = 100 - budget_pct
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Users Targeted", f"{budget_pct}%")
    with col2:
        st.metric("Conversions Captured", f"{pct_of_max:.0f}%")
    with col3:
        st.metric("Cost Savings", f"{cost_savings}%")
    
    st.write("")
    st.info("**In my mind:** The curve flattens after ~20%. Beyond that, you're paying for diminishing returns.")
    
    st.write("")
    with st.expander("Other Use Cases of Uplift Modeling", expanded=False):
        st.markdown("""
        Beyond ad targeting: churn reduction, upselling, offer optimization.
        
        [Read more (Vidora/mParticle)](https://www.vidora.com/ml-in-business/uplift-modeling-some-practical-examples/)
        """)


def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["Background", "The Data Story", "From ATE to CATE", "Model Evaluation", "Customer Segmentation"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        st.markdown("""
**Data Source:**
- [Criteo Uplift Dataset](https://huggingface.co/datasets/criteo/criteo-uplift)
- ~14M samples from RCT
""")
        
        st.divider()
        
        st.markdown("Built by [Sivagugan Jayachandran](https://www.linkedin.com/in/sivagugan-jayachandran/)")
    
    # Render selected page
    if page == "Background":
        render_background_tab()
    elif page == "The Data Story":
        render_data_story_tab()
    elif page == "From ATE to CATE":
        render_cate_tab()
    elif page == "Model Evaluation":
        render_model_evaluation_tab()
    elif page == "Customer Segmentation":
        render_customer_segmentation_tab()


if __name__ == "__main__":
    main()
