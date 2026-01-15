"""
Technical Details Page
Methodology, model specifications, and statistical details for data science team.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PAGE_CONFIG, COLORS
from utils.data_loader import (
    load_model_metrics,
    load_bootstrap_results,
    load_calibration_data,
    load_permutation_importance,
    load_shap_importance,
    load_feature_statistics,
    load_eda_summary
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
    .code-block {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("Technical Details")
    st.markdown("*Methodology, model specifications, and statistical details for data science team*")
    
    # Load data
    model_metrics = load_model_metrics()
    bootstrap_results = load_bootstrap_results()
    calibration_data = load_calibration_data()
    permutation_importance = load_permutation_importance()
    shap_importance = load_shap_importance()
    eda_summary = load_eda_summary()
    
    # Try to load feature statistics
    try:
        feature_stats = load_feature_statistics()
    except:
        feature_stats = None
    
    # Dataset overview
    st.markdown('<p class="section-header">Dataset Specification</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Source:** Criteo Uplift Modeling Dataset  
        **Provider:** Criteo AI Lab via Hugging Face  
        **Type:** Randomized Controlled Trial (RCT)  
        **Domain:** Digital Advertising  
        """)
        
        # Extract values from EDA summary
        def get_metric(metric_name):
            row = eda_summary[eda_summary['Metric'] == metric_name]
            return row['Value'].values[0] if len(row) > 0 else None
        
        total_samples = get_metric('Total Samples')
        control_pct = get_metric('Control %')
        treatment_pct = get_metric('Treatment %')
        conversion_rate = get_metric('Conversion Rate %')
        
        dataset_specs = pd.DataFrame({
            'Attribute': ['Total Samples', 'Features', 'Treatment Split', 'Target Variable', 'Baseline Conversion'],
            'Value': [
                f'{total_samples:,.0f}' if total_samples else 'N/A',
                '12 (f0-f11, anonymized)',
                f'{treatment_pct:.0f}% treatment / {control_pct:.0f}% control' if treatment_pct else 'N/A',
                'Conversion (binary)',
                f'~{conversion_rate:.2f}%' if conversion_rate else 'N/A'
            ]
        })
        st.dataframe(dataset_specs, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Why 85% treatment?**
        
        Unlike typical A/B tests where treatment is minimized to reduce risk, this is an 
        advertising dataset where treatment (showing ads) generates revenue. The 15% control 
        holdout is the minimum needed to measure causal effects while maximizing revenue.
        
        **Randomization verification:** Feature distributions overlap between treatment 
        and control groups, confirming proper randomization.
        """)
    
    # Model architectures
    st.markdown('<p class="section-header">Model Architectures</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["T-Learner", "S-Learner", "X-Learner"])
    
    with tab1:
        st.markdown("""
        **T-Learner (Two-Model Learner)**
        
        The T-Learner trains two separate models:
        - `model_c`: Trained on control group only, predicts P(Y=1 | X, T=0)
        - `model_t`: Trained on treatment group only, predicts P(Y=1 | X, T=1)
        
        CATE estimation: `τ(x) = model_t(x) - model_c(x)`
        """)
        
        st.markdown("""
        ```python
        # T-Learner Implementation
        from xgboost import XGBClassifier
        
        # Split data by treatment
        X_c, y_c = X[T == 0], y[T == 0]
        X_t, y_t = X[T == 1], y[T == 1]
        
        # Train separate models
        model_c = XGBClassifier(n_estimators=100, max_depth=5)
        model_t = XGBClassifier(n_estimators=100, max_depth=5)
        model_c.fit(X_c, y_c)
        model_t.fit(X_t, y_t)
        
        # Predict CATE
        cate = model_t.predict_proba(X)[:, 1] - model_c.predict_proba(X)[:, 1]
        ```
        """)
        
        st.markdown("""
        **Pros:** Simple, interpretable, works well with sufficient samples in both groups  
        **Cons:** Requires enough samples in control group; no regularization toward ATE
        """)
    
    with tab2:
        st.markdown("""
        **S-Learner (Single-Model Learner)**
        
        The S-Learner trains a single model with treatment as an additional feature:
        - Model predicts P(Y=1 | X, T)
        
        CATE estimation: `τ(x) = model(x, T=1) - model(x, T=0)`
        """)
        
        st.markdown("""
        ```python
        # S-Learner Implementation
        from xgboost import XGBClassifier
        
        # Add treatment as feature
        X_with_t = np.column_stack([X, T])
        
        # Train single model
        model = XGBClassifier(n_estimators=100, max_depth=5)
        model.fit(X_with_t, y)
        
        # Predict CATE
        X_t1 = np.column_stack([X, np.ones(len(X))])
        X_t0 = np.column_stack([X, np.zeros(len(X))])
        cate = model.predict_proba(X_t1)[:, 1] - model.predict_proba(X_t0)[:, 1]
        ```
        """)
        
        st.markdown("""
        **Pros:** Simpler than T-Learner; regularizes toward zero treatment effect  
        **Cons:** Treatment effect can be dominated by main effects; may underestimate heterogeneity
        """)
    
    with tab3:
        st.markdown("""
        **X-Learner (Cross-Learner)**
        
        Four-stage approach designed for imbalanced treatment/control splits:
        
        1. **Stage 1:** Train outcome models (same as T-Learner)
        2. **Stage 2:** Impute counterfactual outcomes
        3. **Stage 3:** Train CATE models on imputed treatment effects
        4. **Stage 4:** Combine estimates using propensity weighting
        """)
        
        st.markdown("""
        ```python
        # X-Learner Implementation (simplified)
        from causalml.inference.meta import BaseXClassifier
        
        x_learner = BaseXClassifier(
            learner=XGBClassifier(n_estimators=100, max_depth=5),
            control_outcome_learner=XGBClassifier(),
            treatment_outcome_learner=XGBClassifier(),
            control_effect_learner=XGBRegressor(),
            treatment_effect_learner=XGBRegressor()
        )
        
        x_learner.fit(X, T, y)
        cate = x_learner.predict(X)
        ```
        """)
        
        st.markdown("""
        **Pros:** Designed for imbalanced splits; uses all data efficiently  
        **Cons:** Complex; error propagation through stages; sensitive to propensity estimation
        """)
    
    # Hyperparameters
    st.markdown('<p class="section-header">Model Hyperparameters</p>', unsafe_allow_html=True)
    
    hyperparams = pd.DataFrame({
        'Parameter': ['n_estimators', 'max_depth', 'learning_rate', 'random_state', 'base_score'],
        'Value': ['100', '5', '0.1', '42', '0.5'],
        'Notes': [
            'Number of boosting rounds',
            'Maximum tree depth (regularization)',
            'Step size shrinkage',
            'For reproducibility',
            'Required for SHAP compatibility'
        ]
    })
    st.dataframe(hyperparams, hide_index=True, use_container_width=True)
    
    # Evaluation metrics
    st.markdown('<p class="section-header">Evaluation Metrics</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Qini Coefficient**
    
    The Qini coefficient measures the area between the model's Qini curve and random targeting:
    """)
    
    st.latex(r'''
    Q = \int_0^1 \left[ q(t) - t \cdot q(1) \right] dt
    ''')
    
    st.markdown("""
    where `q(t)` is the cumulative uplift when targeting fraction `t` of the population.
    
    **AUUC (Area Under Uplift Curve)**
    
    The average uplift across all targeting thresholds:
    """)
    
    st.latex(r'''
    AUUC = \int_0^1 \tau(t) \, dt
    ''')
    
    st.markdown("""
    where `τ(t)` is the average uplift for the top `t` fraction of users.
    """)
    
    # Model results
    st.markdown("**Model Performance Comparison:**")
    
    display_metrics = model_metrics.copy()
    display_metrics['Qini_Coefficient'] = display_metrics['Qini_Coefficient'].round(2)
    display_metrics['AUUC'] = display_metrics['AUUC'].apply(lambda x: f'{x:.6f}')
    display_metrics['Mean_Predicted_Uplift'] = display_metrics['Mean_Predicted_Uplift'].apply(lambda x: f'{x:.6f}')
    display_metrics.columns = ['Model', 'Qini Coefficient', 'AUUC', 'Mean Predicted Uplift']
    
    st.dataframe(display_metrics, hide_index=True, use_container_width=True)
    
    # Bootstrap confidence intervals
    st.markdown('<p class="section-header">Statistical Uncertainty</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Bootstrap analysis provides uncertainty estimates for model comparisons. 
    The table below shows pairwise comparisons between models.
    """)
    
    if not bootstrap_results.empty:
        # Bootstrap results has comparison data, not raw bootstrap samples
        st.markdown("**Model Comparison Results:**")
        
        display_bootstrap = bootstrap_results.copy()
        display_bootstrap['Mean_Difference'] = display_bootstrap['Mean_Difference'].apply(lambda x: f'{x:.2f}')
        display_bootstrap['Std_Error'] = display_bootstrap['Std_Error'].apply(lambda x: f'{x:.2f}')
        display_bootstrap['CI_Lower'] = display_bootstrap['CI_Lower'].apply(lambda x: f'{x:.2f}')
        display_bootstrap['CI_Upper'] = display_bootstrap['CI_Upper'].apply(lambda x: f'{x:.2f}')
        display_bootstrap['P_Value'] = display_bootstrap['P_Value'].apply(lambda x: f'{x:.3f}')
        
        st.dataframe(display_bootstrap, hide_index=True, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Interpretation:</strong> While T-Learner shows higher Qini scores, the 
        confidence intervals are wide, indicating substantial uncertainty in the exact 
        performance difference. This is common in uplift modeling where the signal is 
        inherently noisy.
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown('<p class="section-header">Feature Importance Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Permutation Importance**")
        st.markdown("""
        Measures the decrease in model performance when a feature is randomly shuffled. 
        Higher values indicate more important features.
        """)
        
        if not permutation_importance.empty:
            fig_perm = go.Figure()
            
            sorted_perm = permutation_importance.sort_values('Importance', ascending=True).tail(12)
            
            fig_perm.add_trace(go.Bar(
                y=sorted_perm['Feature'],
                x=sorted_perm['Importance'],
                orientation='h',
                marker_color='#1565C0'
            ))
            
            fig_perm.update_layout(
                title="Permutation Importance",
                xaxis_title="Importance",
                height=400
            )
            
            st.plotly_chart(fig_perm, use_container_width=True)
    
    with col2:
        st.markdown("**SHAP Importance**")
        st.markdown("""
        Mean absolute SHAP values measure each feature's average contribution to 
        uplift predictions. Based on the T-Learner model.
        """)
        
        if not shap_importance.empty:
            fig_shap = go.Figure()
            
            sorted_shap = shap_importance.sort_values('Importance', ascending=True).tail(12)
            
            fig_shap.add_trace(go.Bar(
                y=sorted_shap['Feature'],
                x=sorted_shap['Importance'],
                orientation='h',
                marker_color='#2E7D32'
            ))
            
            fig_shap.update_layout(
                title="SHAP Importance for Uplift",
                xaxis_title="Mean |SHAP Value|",
                height=400
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
    
    # Calibration
    st.markdown('<p class="section-header">Model Calibration</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Calibration measures whether predicted uplift values align with observed uplift. 
    We bin predictions and compare mean predicted vs actual uplift in each bin.
    """)
    
    if not calibration_data.empty:
        fig_cal = go.Figure()
        
        # Note: column names from the CSV
        models = calibration_data['Model'].unique()
        colors = {'T-Learner': '#2E7D32', 'S-Learner': '#1565C0', 'X-Learner': '#FF6F00'}
        
        for model in models:
            model_data = calibration_data[calibration_data['Model'] == model]
            
            fig_cal.add_trace(go.Scatter(
                x=model_data['mean_predicted'],
                y=model_data['actual_uplift'],
                mode='lines+markers',
                name=model,
                line=dict(color=colors.get(model, '#666'))
            ))
        
        # Add perfect calibration line
        min_val = calibration_data['mean_predicted'].min()
        max_val = calibration_data['mean_predicted'].max()
        fig_cal.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash')
        ))
        
        fig_cal.update_layout(
            title="Calibration: Predicted vs Actual Uplift",
            xaxis_title="Mean Predicted Uplift",
            yaxis_title="Actual Uplift",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig_cal, use_container_width=True)
    
    # Segmentation methodology
    st.markdown('<p class="section-header">Segmentation Methodology</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Customers are segmented into four groups based on predicted uplift and baseline 
    conversion probability:
    """)
    
    segmentation_logic = pd.DataFrame({
        'Segment': ['Persuadables', 'Sleeping Dogs', 'Sure Things', 'Lost Causes'],
        'Criteria': [
            'Uplift >= 75th percentile',
            'Uplift < 0',
            'Uplift in [0, 75th percentile] AND baseline >= median',
            'Uplift in [0, 75th percentile] AND baseline < median'
        ],
        'Interpretation': [
            'High treatment effect - convert because of ad',
            'Negative treatment effect - ad hurts conversion',
            'Low effect, high baseline - would convert anyway',
            'Low effect, low baseline - will not convert regardless'
        ]
    })
    
    st.dataframe(segmentation_logic, hide_index=True, use_container_width=True)
    
    st.markdown("""
    ```python
    def segment_customers(uplift, baseline_prob, percentile_threshold=0.75):
        high_uplift_cutoff = np.percentile(uplift, percentile_threshold * 100)
        median_baseline = np.median(baseline_prob)
        
        if uplift < 0:
            return 'Sleeping Dogs'
        elif uplift >= high_uplift_cutoff:
            return 'Persuadables'
        elif baseline_prob >= median_baseline:
            return 'Sure Things'
        else:
            return 'Lost Causes'
    ```
    """)
    
    # Limitations and assumptions
    st.markdown('<p class="section-header">Limitations and Assumptions</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Key Assumptions:**
    
    1. **SUTVA (Stable Unit Treatment Value Assumption):** One user's treatment does not 
       affect another user's outcome. This may not hold if there are network effects.
    
    2. **Unconfoundedness:** Given observed features, treatment assignment is independent 
       of potential outcomes. This holds by design in an RCT.
    
    3. **Positivity:** All users have positive probability of receiving treatment. 
       Verified by the 85/15 treatment split.
    
    **Limitations:**
    
    1. **Feature anonymization:** We cannot interpret feature importance in business terms.
    
    2. **Single time period:** The analysis is cross-sectional; user behavior may change 
       over time.
    
    3. **Binary treatment:** We only compare ad vs. no ad; does not capture intensity 
       or creative variations.
    
    4. **Model uncertainty:** Point predictions do not capture full uncertainty; bootstrap 
       intervals provide partial mitigation.
    
    **Recommendations for Production:**
    
    1. Implement holdout validation before scaling targeting
    2. Monitor model drift over time
    3. Consider ensemble of models for robustness
    4. A/B test model-based targeting vs. current approach
    """)
    
    # References
    st.markdown('<p class="section-header">References</p>', unsafe_allow_html=True)
    
    st.markdown("""
    1. **Kunzel, S. R., et al. (2019).** "Metalearners for Estimating Heterogeneous 
       Treatment Effects using Machine Learning." PNAS.
    
    2. **Radcliffe, N. J., & Surry, P. D. (2011).** "Real-World Uplift Modelling with 
       Significance-Based Uplift Trees." Stochastic Solutions.
    
    3. **Gutierrez, P., & Gerardy, J. Y. (2017).** "Causal Inference and Uplift Modeling: 
       A Review of the Literature." JMLR Workshop and Conference Proceedings.
    
    4. **CausalML Documentation:** [github.com/uber/causalml](https://github.com/uber/causalml)
    
    5. **Criteo Dataset:** [huggingface.co/datasets/criteo/criteo-uplift](https://huggingface.co/datasets/criteo/criteo-uplift)
    """)


if __name__ == "__main__":
    main()
