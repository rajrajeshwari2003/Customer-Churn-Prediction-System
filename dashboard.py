import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import joblib

# ========================================================
# Phase 10: Streamlit Interactive Retention Dashboard 
# Premium UI/UX Upgrade
# ========================================================

st.set_page_config(
    page_title="Enterprise Churn Dashboard", 
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render Custom CSS for premium Glassmorphism styling and smooth fonts
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #4F8BF9;
        margin-bottom: 2px;
    }
    .sub-head {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Clean minimalist headers
st.markdown("<h1 class='main-header'>🔮 AI-Powered Customer Retention Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-head'>Identify high-risk customer turnover and simulate retention scenarios using Explainable AI globally.</div>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_resource
def load_predictive_engine():
    try:
        return joblib.load("best_churn_model.pkl")
    except Exception:
        return None

def main():
    model = load_predictive_engine()
    
    # ----------------- CLEAN PREMIUM SIDEBAR -----------------
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2809/2809071.png", width=60)
    st.sidebar.header("Data Ingestion")
    st.sidebar.info("Upload fresh, unlabeled telemetry vectors to sweep for flight risks.")
    
    uploaded_file = st.sidebar.file_uploader("📥 Upload CSV Telemetry", type=["csv"])
    
    # ==========================================
    # EMPTY LANDING STATE (NO FILE)
    # ==========================================
    if uploaded_file is None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info("👈 **Awaiting Data Ingestion**\n\nPlease upload a target Telemetry CSV file from the sidebar menu to awaken the AI engine.")
        
        with col2:
            if model is not None:
                st.success("✅ **Core Engine Online**: `best_churn_model.pkl` heavily loaded into memory and standing by.")
            else:
                st.error("❌ **Core Engine Offline**: `best_churn_model.pkl` not found in directory. Train your pipeline first!")
                
        st.markdown("<br><br>### Expected Output Architecture", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Profiles Scanned", "--")
        m2.metric("High-Risk Retention Targets", "--")
        m3.metric("Projected Revenue at Risk", "$--")
        
    # ==========================================
    # ACTIVE DASHBOARD STATE (FILE UPLOADED)
    # ==========================================
    else:
        df = pd.read_csv(uploaded_file)
        
        # Streamlit Tabs for Clean Logical UI Separation
        tab_summary, tab_analytics, tab_shap, tab_raw = st.tabs([
            "📊 Executive Dashboard", 
            "🚨 Priority Target List", 
            "🧠 Deep AI Explainability", 
            "📄 Raw Ingestion Matrix"
        ])
        
        # ----- TAB 4: Raw Data Inspector -----
        with tab_raw:
            st.markdown("### Raw Telemetry Inspector")
            st.dataframe(df.head(50), use_container_width=True)
            
        with st.spinner('Engaging Deep Scikit-Learn Predictive Sweep...'):
            if model is not None:
                probabilities = model.predict_proba(df)[:, 1]
                df['Churn_Probability'] = probabilities
                
                if 'MonthlyCharges' in df.columns:
                    df['Retention_Priority_Score'] = df['Churn_Probability'] * df['MonthlyCharges']
                else:
                    df['Retention_Priority_Score'] = df['Churn_Probability'] * 100 
                
                # Logic Bounds
                high_risk_threshold = 0.50
                high_risk_count = len(df[df['Churn_Probability'] >= high_risk_threshold])
                total_risk_mrr = df[df['Churn_Probability'] >= high_risk_threshold]['Retention_Priority_Score'].sum()
                
                # ----- TAB 1: EXECUTIVE DASHBOARD -----
                with tab_summary:
                    st.markdown("### Fleet Trajectory Summary")
                    
                    # Beautiful metrics utilizing st.metric layout natively
                    c1, c2, c3 = st.columns(3)
                    with c1:
                         st.metric("Total Customers Scanned", f"{len(df):,}")
                    with c2:
                         st.metric("High-Risk Flight Candidates", f"{high_risk_count:,}")
                    with c3:
                         st.metric("Estimated MRR Risk Exposure", f"₹{total_risk_mrr:,.2f}")
                    
                    st.divider()
                    
                    # Histogram logic beautifully plotting distribution curves
                    st.markdown("#### Probability Risk Distribution Curve")
                    fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
                    ax_hist.hist(df['Churn_Probability'], bins=25, color='#4F8BF9', alpha=0.8, edgecolor='white')
                    ax_hist.axvline(high_risk_threshold, color='red', linestyle='dashed', linewidth=2, label='Critical Threshold Boundary')
                    ax_hist.set_xlabel("Algorithmic Churn Probability")
                    ax_hist.set_ylabel("Customer Volume")
                    ax_hist.spines['top'].set_visible(False)
                    ax_hist.spines['right'].set_visible(False)
                    ax_hist.legend()
                    st.pyplot(fig_hist)
                
                # ----- TAB 2: PRIORITY TARGET LIST -----
                with tab_analytics:
                    st.markdown("### 🔥 Immediate Intervention Queue")
                    st.markdown("Target these individuals precisely with retention campaigns. Sorted maximally by Total Possible MRR Loss.")
                    
                    at_risk_df = df.sort_values(by='Retention_Priority_Score', ascending=False)
                    
                    display_cols = ['customerID', 'Churn_Probability', 'Retention_Priority_Score']
                    if 'MonthlyCharges' in df.columns: display_cols.append('MonthlyCharges')
                    if 'Contract' in df.columns: display_cols.append('Contract')
                    if 'tenure' in df.columns: display_cols.append('tenure')
                        
                    display_cols = [c for c in display_cols if c in at_risk_df.columns]
                    
                    styled_df = at_risk_df[display_cols].head(50).style.background_gradient(cmap='Reds', subset=['Retention_Priority_Score'])
                    st.dataframe(styled_df, use_container_width=True, height=600)
                    
                # ----- TAB 3: SHAP DEEP EXPLAINABILITY -----
                with tab_shap:
                    st.markdown("### 🧠 Interpretability Engine")
                    st.markdown("Translating pure mathematical abstractions into highly human-readable, global business drivers.")
                    
                    try:
                        classifier = model.named_steps['classifier']
                        # Transform data using all steps prior to the classifier
                        preprocessor_pipeline = model[:-1]
                        X_trans = preprocessor_pipeline.transform(df)
                        
                        try:
                            encoder = model.named_steps['encoding_and_scaling']
                            feature_names = encoder.get_feature_names_out()
                        except:
                            feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
                        
                        df_transformed = pd.DataFrame(X_trans, columns=feature_names)
                        
                        explainer = shap.TreeExplainer(classifier)
                        shap_values = explainer(df_transformed)
                        
                        if len(shap_values.shape) == 3:
                            shap_values = shap_values[:, :, 1]
                            
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, show=False)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"Engine Failed SHAP isolation natively due to lack of model dimensionality: {e}")
            else:
                st.error("Engine offline cleanly. Save your trained scikit estimator pipeline exactly as 'best_churn_model.pkl'.")

if __name__ == "__main__":
    main()
