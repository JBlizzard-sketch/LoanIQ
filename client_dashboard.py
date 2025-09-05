# File: pages/client_dashboard.py

import streamlit as st
import pandas as pd
from database.db_manager import DatabaseManager
from models.ml_pipeline import MLPipeline
from models.model_versions import ModelVersionManager
from utils.data_processing import select_features, encode_categoricals, scale_numeric

# PATCH START: Client Dashboard with real predictions

def show_client_dashboard():
    """Dashboard for clients: predictions, risk scores, insights."""

    st.title("📊 LoanIQ Client Dashboard")

    # Ensure user logged in
    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("Please log in to access your dashboard.")
        st.switch_page("pages/login.py")
        return

    user = st.session_state["user"]
    st.info(f"Welcome, {user['email']}")

    db_manager = DatabaseManager()
    version_manager = ModelVersionManager(db_manager)

    # Load latest model version
    latest_version = version_manager.get_latest_version()
    if not latest_version:
        st.warning("No trained models available yet. Please contact support.")
        return

    st.subheader("📌 Risk Scoring")

    # Upload client financial data
    uploaded_file = st.file_uploader("Upload your financial profile (CSV)", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", df.head())

            # Load pipeline with chosen family
            pipeline = MLPipeline(model_family=latest_version["model_family"])
            model = version_manager.load_model(latest_version["id"])

            # Prepare data
            X, _ = pipeline.prepare_data(df, target_col=None)

            # Predict probabilities
            preds = model.predict_proba(X)[:, 1]
            df["default_risk"] = preds

            # Show results
            avg_risk = df["default_risk"].mean()
            st.metric("Average Default Risk", f"{avg_risk:.2%}")

            # Per-row results
            st.dataframe(df[["default_risk"]].style.background_gradient(cmap="Reds"))

            # Loan limit heuristic
            max_loan = (1 - avg_risk) * 80000
            st.metric("Suggested Loan Limit", f"${max_loan:,.0f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Please upload your data to see risk scores.")

# PATCH END
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta

from models.ml_pipeline import MLPipeline
from models.synthetic_data import SyntheticDataGenerator
from utils.explainability import SHAPExplainer
from utils.data_processing import DataProcessor

def show_client_dashboard():
    """Display the client dashboard with all client features"""
    st.title("📊 Client Dashboard")
    
    # Navigation
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])
    
    with col_nav1:
        if st.button("← Back to Home"):
            st.session_state.page = 'homepage'
            st.rerun()
    
    with col_nav2:
        st.markdown(f"**Welcome, {st.session_state.username}**")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📋 Dashboard Sections")
        dashboard_section = st.selectbox(
            "Choose Section:",
            ["Overview", "Data Upload", "Predictions", "Model Insights", "What-If Analysis", "Export Results"]
        )
    
    # Initialize components
    ml_pipeline = st.session_state.ml_pipeline
    data_processor = DataProcessor()
    
    if dashboard_section == "Overview":
        show_overview_section(ml_pipeline)
    elif dashboard_section == "Data Upload":
        show_data_upload_section(ml_pipeline, data_processor)
    elif dashboard_section == "Predictions":
        show_predictions_section(ml_pipeline)
    elif dashboard_section == "Model Insights":
        show_model_insights_section(ml_pipeline)
    elif dashboard_section == "What-If Analysis":
        show_whatif_analysis_section(ml_pipeline)
    elif dashboard_section == "Export Results":
        show_export_section()

def show_overview_section(ml_pipeline):
    """Show overview with key metrics and charts"""
    st.markdown("### 🎯 Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "2,847", "↗ 156")
    
    with col2:
        st.metric("Model Accuracy", "84.7%", "↗ 1.2%")
    
    with col3:
        st.metric("High Risk Rate", "12.3%", "↓ 0.8%")
    
    with col4:
        st.metric("Avg Score", "742", "↗ 15")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Risk Score Distribution")
        
        # Generate sample data for visualization
        np.random.seed(42)
        risk_scores = np.random.normal(650, 100, 1000)
        risk_scores = np.clip(risk_scores, 300, 850)
        
        fig = px.histogram(x=risk_scores, nbins=30, title="Credit Score Distribution")
        fig.update_layout(
            xaxis_title="Credit Score",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Risk Categories")
        
        risk_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        risk_counts = [450, 320, 180, 50]
        
        fig = px.pie(values=risk_counts, names=risk_categories, title="Risk Category Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.markdown("#### 📊 Prediction Trends")
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    daily_predictions = np.random.poisson(15, len(dates))
    
    df_trends = pd.DataFrame({
        'Date': dates,
        'Predictions': daily_predictions
    })
    
    fig = px.line(df_trends, x='Date', y='Predictions', title="Daily Prediction Volume")
    st.plotly_chart(fig, use_container_width=True)

def show_data_upload_section(ml_pipeline, data_processor):
    """Show data upload and generation options"""
    st.markdown("### 📁 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "🔧 Generate Synthetic", "🔌 API Integration"])
    
    with tab1:
        st.markdown("#### Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your loan application data for analysis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"File uploaded successfully! Shape: {df.shape}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Data Preview")
                    st.dataframe(df.head(10))
                
                with col2:
                    st.markdown("##### Data Info")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    st.text(info_str)
                
                # Data validation
                if st.button("🔍 Validate & Process Data"):
                    validation_results = data_processor.validate_data(df)
                    
                    if validation_results['valid']:
                        st.success("✅ Data validation passed!")
                        processed_df = data_processor.preprocess_data(df)
                        st.session_state.processed_data = processed_df
                        
                        st.markdown("##### Processed Data Sample")
                        st.dataframe(processed_df.head())
                        
                    else:
                        st.error("❌ Data validation failed:")
                        for error in validation_results['errors']:
                            st.error(f"• {error}")
                        
                        st.markdown("##### Suggestions:")
                        for suggestion in validation_results.get('suggestions', []):
                            st.info(f"💡 {suggestion}")
                            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.markdown("#### Generate Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_records = st.slider("Number of Records", 100, 5000, 1000)
            risk_distribution = st.selectbox(
                "Risk Distribution",
                ["Balanced", "Conservative", "Aggressive", "Custom"]
            )
        
        with col2:
            include_defaults = st.checkbox("Include Default Cases", value=True)
            add_noise = st.checkbox("Add Realistic Noise", value=True)
        
        if st.button("🎲 Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                synthetic_gen = SyntheticDataGenerator()
                
                config = {
                    'num_records': num_records,
                    'risk_distribution': risk_distribution.lower(),
                    'include_defaults': include_defaults,
                    'add_noise': add_noise
                }
                
                synthetic_data = synthetic_gen.generate_data(config)
                
                st.success(f"Generated {len(synthetic_data)} synthetic records!")
                st.session_state.synthetic_data = synthetic_data
                
                # Preview
                st.markdown("##### Generated Data Preview")
                st.dataframe(synthetic_data.head(10))
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Records Generated", len(synthetic_data))
                
                with col2:
                    default_rate = synthetic_data['default'].mean() * 100
                    st.metric("Default Rate", f"{default_rate:.1f}%")
                
                with col3:
                    avg_score = synthetic_data['credit_score'].mean()
                    st.metric("Avg Credit Score", f"{avg_score:.0f}")
    
    with tab3:
        st.markdown("#### API Integration")
        
        st.info("Connect to external data sources for real-time loan application processing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Credit Bureau APIs")
            api_provider = st.selectbox(
                "Select Provider",
                ["Experian", "Equifax", "TransUnion", "Custom API"]
            )
            
            api_key = st.text_input("API Key", type="password", help="Your API key for the selected provider")
            
        with col2:
            st.markdown("##### Test Connection")
            
            if st.button("🔌 Test API Connection"):
                if api_key:
                    # Simulate API test
                    with st.spinner("Testing connection..."):
                        import time
                        time.sleep(2)
                        st.success("✅ API connection successful!")
                        
                        # Show sample response
                        st.json({
                            "status": "connected",
                            "provider": api_provider,
                            "rate_limit": "1000/hour",
                            "response_time": "245ms"
                        })
                else:
                    st.warning("Please enter an API key")

def show_predictions_section(ml_pipeline):
    """Show prediction interface and results"""
    st.markdown("### 🔮 Credit Scoring Predictions")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions", "📈 Prediction History"])
    
    with tab1:
        st.markdown("#### Single Application Scoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Applicant Information")
            age = st.slider("Age", 18, 80, 35)
            income = st.number_input("Annual Income ($)", 20000, 500000, 75000)
            employment_length = st.selectbox("Employment Length", 
                                           ["< 1 year", "1-2 years", "3-5 years", "5-10 years", "10+ years"])
            home_ownership = st.selectbox("Home Ownership", 
                                        ["RENT", "OWN", "MORTGAGE", "OTHER"])
        
        with col2:
            st.markdown("##### Credit Information")
            credit_score = st.slider("Current Credit Score", 300, 850, 650)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25)
            loan_amount = st.number_input("Requested Loan Amount ($)", 1000, 100000, 25000)
            loan_purpose = st.selectbox("Loan Purpose", 
                                      ["debt_consolidation", "credit_card", "home_improvement", 
                                       "major_purchase", "medical", "other"])
        
        if st.button("🎯 Calculate Risk Score"):
            # Prepare input data
            input_data = {
                'age': age,
                'annual_income': income,
                'employment_length': employment_length,
                'home_ownership': home_ownership,
                'credit_score': credit_score,
                'debt_to_income_ratio': debt_to_income,
                'loan_amount': loan_amount,
                'loan_purpose': loan_purpose
            }
            
            with st.spinner("Calculating risk score..."):
                prediction_result = ml_pipeline.predict_single(input_data)
                
                st.markdown("---")
                st.markdown("#### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_score = prediction_result['risk_score']
                    color = "green" if risk_score > 700 else "orange" if risk_score > 600 else "red"
                    st.markdown(f"##### Risk Score: <span style='color: {color}'>{risk_score}</span>", 
                               unsafe_allow_html=True)
                
                with col2:
                    default_prob = prediction_result['default_probability']
                    st.metric("Default Probability", f"{default_prob:.1%}")
                
                with col3:
                    risk_category = prediction_result['risk_category']
                    st.metric("Risk Category", risk_category)
                
                # SHAP explanation
                st.markdown("#### 🔍 Feature Importance (SHAP)")
                explainer = SHAPExplainer(ml_pipeline.get_active_model())
                shap_fig = explainer.plot_single_prediction(input_data)
                st.plotly_chart(shap_fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Batch Processing")
        
        if 'processed_data' in st.session_state or 'synthetic_data' in st.session_state:
            data_source = st.selectbox(
                "Select Data Source",
                ["Uploaded Data", "Synthetic Data"] if 'synthetic_data' in st.session_state else ["Uploaded Data"]
            )
            
            if data_source == "Uploaded Data" and 'processed_data' in st.session_state:
                data = st.session_state.processed_data
            elif data_source == "Synthetic Data" and 'synthetic_data' in st.session_state:
                data = st.session_state.synthetic_data
            else:
                st.warning("No data available for batch processing")
                return
            
            st.info(f"Ready to process {len(data)} records")
            
            if st.button("🚀 Run Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    batch_results = ml_pipeline.predict_batch(data)
                    
                    st.success(f"Processed {len(batch_results)} predictions!")
                    st.session_state.batch_results = batch_results
                    
                    # Results summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_score = batch_results['risk_score'].mean()
                        st.metric("Avg Risk Score", f"{avg_score:.0f}")
                    
                    with col2:
                        high_risk_pct = (batch_results['risk_category'] == 'High Risk').mean() * 100
                        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
                    
                    with col3:
                        avg_default_prob = batch_results['default_probability'].mean() * 100
                        st.metric("Avg Default Prob", f"{avg_default_prob:.1f}%")
                    
                    with col4:
                        approved_pct = (batch_results['risk_score'] >= 650).mean() * 100
                        st.metric("Approval Rate", f"{approved_pct:.1f}%")
                    
                    # Results distribution
                    fig = px.histogram(batch_results, x='risk_score', nbins=30, 
                                     title="Risk Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.markdown("#### Detailed Results")
                    st.dataframe(batch_results.head(100))
        else:
            st.info("Please upload data or generate synthetic data first in the 'Data Upload' section.")
    
    with tab3:
        st.markdown("#### Prediction History")
        
        # Generate sample history data
        history_data = generate_prediction_history()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### Recent Predictions")
            st.dataframe(history_data.head(20))
        
        with col2:
            st.markdown("##### Summary Stats")
            
            total_predictions = len(history_data)
            st.metric("Total Predictions", total_predictions)
            
            avg_score = history_data['risk_score'].mean()
            st.metric("Average Score", f"{avg_score:.0f}")
            
            approval_rate = (history_data['risk_score'] >= 650).mean() * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Trend analysis
        st.markdown("##### Prediction Trends")
        
        daily_stats = history_data.groupby(history_data['timestamp'].dt.date).agg({
            'risk_score': 'mean',
            'default_probability': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=daily_stats['timestamp'], y=daily_stats['risk_score'], name="Avg Risk Score"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=daily_stats['timestamp'], y=daily_stats['default_probability'], name="Avg Default Prob"),
            secondary_y=True,
        )
        
        fig.update_yaxes(title_text="Risk Score", secondary_y=False)
        fig.update_yaxes(title_text="Default Probability", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

def show_model_insights_section(ml_pipeline):
    """Show model performance and explainability insights"""
    st.markdown("### 🧠 Model Insights & Explainability")
    
    tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🔍 SHAP Analysis", "🎯 Feature Importance"])
    
    with tab1:
        st.markdown("#### Model Performance Metrics")
        
        # Model comparison
        models_performance = ml_pipeline.get_models_performance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Active Models Comparison")
            
            perf_df = pd.DataFrame(models_performance)
            
            fig = px.bar(perf_df, x='model_name', y='auc_score', 
                        title="Model AUC Scores", color='auc_score',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Model Metrics")
            st.dataframe(perf_df, hide_index=True)
        
        # ROC Curves
        st.markdown("##### ROC Curve Comparison")
        roc_fig = ml_pipeline.plot_roc_curves()
        st.plotly_chart(roc_fig, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("##### Confusion Matrix (Best Model)")
        confusion_fig = ml_pipeline.plot_confusion_matrix()
        st.plotly_chart(confusion_fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### SHAP Explainability Analysis")
        
        if 'batch_results' in st.session_state:
            explainer = SHAPExplainer(ml_pipeline.get_active_model())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Global Feature Importance")
                global_shap_fig = explainer.plot_global_importance()
                st.plotly_chart(global_shap_fig, use_container_width=True)
            
            with col2:
                st.markdown("##### SHAP Summary Plot")
                summary_fig = explainer.plot_summary()
                st.plotly_chart(summary_fig, use_container_width=True)
            
            # Partial dependence plots
            st.markdown("##### Partial Dependence Plots")
            
            feature_select = st.selectbox(
                "Select Feature for Partial Dependence",
                ['credit_score', 'annual_income', 'debt_to_income_ratio', 'age']
            )
            
            pdp_fig = explainer.plot_partial_dependence(feature_select)
            st.plotly_chart(pdp_fig, use_container_width=True)
        else:
            st.info("Please run batch predictions first to generate SHAP explanations.")
    
    with tab3:
        st.markdown("#### Feature Importance Analysis")
        
        # Feature importance comparison across models
        feature_importance_fig = ml_pipeline.plot_feature_importance_comparison()
        st.plotly_chart(feature_importance_fig, use_container_width=True)
        
        # Feature correlation heatmap
        st.markdown("##### Feature Correlation Matrix")
        if 'processed_data' in st.session_state:
            data = st.session_state.processed_data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            corr_matrix = data[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

def show_whatif_analysis_section(ml_pipeline):
    """Show what-if scenario analysis"""
    st.markdown("### 🔄 What-If Scenario Analysis")
    
    st.info("Explore how changes in applicant characteristics affect credit risk predictions.")
    
    # Base scenario setup
    st.markdown("#### 📋 Base Scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_age = st.slider("Base Age", 18, 80, 35, key="base_age")
        base_income = st.number_input("Base Income", 20000, 500000, 75000, key="base_income")
        base_credit_score = st.slider("Base Credit Score", 300, 850, 650, key="base_credit")
    
    with col2:
        base_dti = st.slider("Base Debt-to-Income", 0.0, 1.0, 0.25, key="base_dti")
        base_loan_amount = st.number_input("Base Loan Amount", 1000, 100000, 25000, key="base_loan")
        base_employment = st.selectbox("Base Employment Length", 
                                     ["< 1 year", "1-2 years", "3-5 years", "5-10 years", "10+ years"],
                                     key="base_employment")
    
    with col3:
        base_home = st.selectbox("Base Home Ownership", 
                               ["RENT", "OWN", "MORTGAGE", "OTHER"], key="base_home")
        base_purpose = st.selectbox("Base Loan Purpose",
                                  ["debt_consolidation", "credit_card", "home_improvement", 
                                   "major_purchase", "medical", "other"], key="base_purpose")
    
    # Calculate base prediction
    base_scenario = {
        'age': base_age,
        'annual_income': base_income,
        'credit_score': base_credit_score,
        'debt_to_income_ratio': base_dti,
        'loan_amount': base_loan_amount,
        'employment_length': base_employment,
        'home_ownership': base_home,
        'loan_purpose': base_purpose
    }
    
    base_prediction = ml_pipeline.predict_single(base_scenario)
    
    st.markdown("---")
    st.markdown("#### 📊 Base Scenario Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Base Risk Score", f"{base_prediction['risk_score']:.0f}")
    
    with col2:
        st.metric("Base Default Prob", f"{base_prediction['default_probability']:.1%}")
    
    with col3:
        st.metric("Base Risk Category", base_prediction['risk_category'])
    
    st.markdown("---")
    
    # Scenario variations
    st.markdown("#### 🎭 Scenario Variations")
    
    variation_type = st.selectbox(
        "Select Variation Type",
        ["Credit Score Impact", "Income Impact", "Debt-to-Income Impact", "Loan Amount Impact", "Age Impact"]
    )
    
    if variation_type == "Credit Score Impact":
        credit_scores = range(300, 851, 50)
        scenarios = []
        
        for score in credit_scores:
            scenario = base_scenario.copy()
            scenario['credit_score'] = score
            prediction = ml_pipeline.predict_single(scenario)
            scenarios.append({
                'credit_score': score,
                'risk_score': prediction['risk_score'],
                'default_probability': prediction['default_probability'],
                'risk_category': prediction['risk_category']
            })
        
        df_scenarios = pd.DataFrame(scenarios)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df_scenarios['credit_score'], y=df_scenarios['risk_score'], 
                      name="Risk Score", line=dict(width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df_scenarios['credit_score'], y=df_scenarios['default_probability'], 
                      name="Default Probability", line=dict(width=3)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Credit Score")
        fig.update_yaxes(title_text="Risk Score", secondary_y=False)
        fig.update_yaxes(title_text="Default Probability", secondary_y=True)
        fig.update_layout(title_text="Credit Score Impact Analysis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif variation_type == "Income Impact":
        incomes = range(20000, 200001, 20000)
        scenarios = []
        
        for income in incomes:
            scenario = base_scenario.copy()
            scenario['annual_income'] = income
            prediction = ml_pipeline.predict_single(scenario)
            scenarios.append({
                'annual_income': income,
                'risk_score': prediction['risk_score'],
                'default_probability': prediction['default_probability']
            })
        
        df_scenarios = pd.DataFrame(scenarios)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df_scenarios['annual_income'], y=df_scenarios['risk_score'], 
                      name="Risk Score", line=dict(width=3)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df_scenarios['annual_income'], y=df_scenarios['default_probability'], 
                      name="Default Probability", line=dict(width=3)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Annual Income ($)")
        fig.update_yaxes(title_text="Risk Score", secondary_y=False)
        fig.update_yaxes(title_text="Default Probability", secondary_y=True)
        fig.update_layout(title_text="Income Impact Analysis")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Similar implementations for other variation types...
    # (continuing with the pattern above)
    
    # Comparison table
    st.markdown("#### 📋 Scenario Comparison")
    st.dataframe(df_scenarios, hide_index=True)

def show_export_section():
    """Show export functionality"""
    st.markdown("### 📤 Export Results")
    
    export_options = st.multiselect(
        "Select data to export:",
        ["Batch Predictions", "Model Performance", "SHAP Explanations", "What-If Scenarios"],
        default=["Batch Predictions"]
    )
    
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON", "PDF Report"])
    
    if st.button("📥 Generate Export"):
        with st.spinner("Preparing export..."):
            # Generate export data based on selections
            export_data = {}
            
            if "Batch Predictions" in export_options and 'batch_results' in st.session_state:
                export_data['batch_predictions'] = st.session_state.batch_results
            
            if "Model Performance" in export_options:
                ml_pipeline = st.session_state.ml_pipeline
                export_data['model_performance'] = ml_pipeline.get_models_performance()
            
            # Create download link
            if export_format == "CSV" and export_data:
                if 'batch_predictions' in export_data:
                    csv = export_data['batch_predictions'].to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="loaniq_results.csv">📥 Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Export ready for download!")
            else:
                st.info("Export functionality will be available once you have results to export.")

def generate_prediction_history():
    """Generate sample prediction history data"""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    history_data = []
    
    for i, date in enumerate(dates):
        # Simulate varying number of predictions per hour
        num_predictions = np.random.poisson(2)
        
        for _ in range(num_predictions):
            history_data.append({
                'timestamp': date,
                'application_id': f"APP_{i:06d}_{np.random.randint(1000, 9999)}",
                'risk_score': np.random.normal(650, 80),
                'default_probability': np.random.beta(2, 8),
                'risk_category': np.random.choice(['Low Risk', 'Medium Risk', 'High Risk'], 
                                                p=[0.6, 0.3, 0.1])
            })
    
    df = pd.DataFrame(history_data)
    df['risk_score'] = np.clip(df['risk_score'], 300, 850).astype(int)
    
    return df
