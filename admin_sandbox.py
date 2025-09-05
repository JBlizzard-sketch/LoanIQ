import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from models.ml_pipeline import MLPipeline
from models.synthetic_data import SyntheticDataGenerator
from models.model_versions import ModelVersionManager
from utils.data_processing import DataProcessor

def show_admin_sandbox():
    """Display the admin sandbox with advanced features"""
    st.title("🔧 Admin Sandbox")
    st.markdown("### Advanced Model Management & System Administration")
    
    # Navigation
    col_nav1, col_nav2 = st.columns([1, 3])
    
    with col_nav1:
        if st.button("← Back to Home"):
            st.session_state.page = 'homepage'
            st.rerun()
    
    with col_nav2:
        st.markdown(f"**Admin Panel - Welcome, {st.session_state.username}**")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🛠️ Admin Sections")
        admin_section = st.selectbox(
            "Choose Section:",
            ["System Overview", "Model Management", "Data Generation", "Stress Testing", 
             "Schema Editor", "User Management", "System Logs"]
        )
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Retrain All Models"):
            retrain_all_models()
        
        if st.button("🧹 Clean System Cache"):
            clean_system_cache()
        
        if st.button("📊 Generate Report"):
            generate_system_report()
    
    # Route to appropriate admin section
    if admin_section == "System Overview":
        show_system_overview()
    elif admin_section == "Model Management":
        show_model_management()
    elif admin_section == "Data Generation":
        show_advanced_data_generation()
    elif admin_section == "Stress Testing":
        show_stress_testing()
    elif admin_section == "Schema Editor":
        show_schema_editor()
    elif admin_section == "User Management":
        show_user_management()
    elif admin_section == "System Logs":
        show_system_logs()

def show_system_overview():
    """Show comprehensive system overview"""
    st.markdown("### 🌐 System Overview")
    
    # System health metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("System Health", "98.5%", "↗ 0.2%")
    
    with col2:
        st.metric("Active Models", "6", "→ 0")
    
    with col3:
        st.metric("API Calls/Day", "12,847", "↗ 1,203")
    
    with col4:
        st.metric("Storage Used", "2.3 GB", "↗ 45 MB")
    
    with col5:
        st.metric("Active Users", "89", "↗ 12")
    
    st.markdown("---")
    
    # System performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Model Performance Trends")
        
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        models = ['XGBoost', 'LightGBM', 'RandomForest', 'LogisticRegression']
        
        performance_data = []
        for date in dates:
            for model in models:
                performance_data.append({
                    'date': date,
                    'model': model,
                    'auc_score': np.random.normal(0.82, 0.02)
                })
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = px.line(df_perf, x='date', y='auc_score', color='model',
                     title="Model AUC Scores Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔄 System Resource Usage")
        
        # Resource usage data
        resources = ['CPU', 'Memory', 'Disk', 'Network']
        usage = [65, 78, 45, 32]
        
        fig = px.bar(x=resources, y=usage, title="Current Resource Usage (%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent system events
    st.markdown("#### 🕐 Recent System Events")
    
    events_data = {
        'Timestamp': [
            '2024-09-04 12:05:23',
            '2024-09-04 11:45:12',
            '2024-09-04 11:30:45',
            '2024-09-04 10:15:33',
            '2024-09-04 09:55:21'
        ],
        'Event Type': ['Model Deployment', 'Data Ingestion', 'User Registration', 'Model Training', 'System Backup'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success', '✅ Success'],
        'Details': [
            'XGBoost v2.3 deployed successfully',
            'Batch data processed: 1,500 records',
            'New client user registered',
            'LightGBM model retrained with new data',
            'Daily system backup completed'
        ]
    }
    
    df_events = pd.DataFrame(events_data)
    st.dataframe(df_events, use_container_width=True, hide_index=True)
    
    # System configuration
    st.markdown("#### ⚙️ System Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Model Settings")
        st.json({
            "default_model": "XGBoost",
            "auto_retrain": True,
            "retrain_threshold": 0.05,
            "max_models": 10
        })
    
    with col2:
        st.markdown("##### Data Settings")
        st.json({
            "batch_size": 1000,
            "validation_split": 0.2,
            "feature_selection": True,
            "data_retention_days": 365
        })
    
    with col3:
        st.markdown("##### API Settings")
        st.json({
            "rate_limit": "1000/hour",
            "timeout": "30s",
            "max_payload": "10MB",
            "authentication": "required"
        })

def show_model_management():
    """Show advanced model management interface"""
    st.markdown("### 🤖 Model Management")
    
    ml_pipeline = st.session_state.ml_pipeline
    version_manager = ModelVersionManager()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Active Models", "🔄 Model Training", "📈 Version Control", "🎯 A/B Testing"])
    
    with tab1:
        st.markdown("#### Currently Active Models")
        
        models_info = ml_pipeline.get_detailed_models_info()
        
        for model_info in models_info:
            with st.expander(f"🤖 {model_info['name']} - v{model_info['version']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Performance Metrics**")
                    st.metric("AUC Score", f"{model_info['auc_score']:.3f}")
                    st.metric("Precision", f"{model_info['precision']:.3f}")
                    st.metric("Recall", f"{model_info['recall']:.3f}")
                
                with col2:
                    st.markdown("**Model Info**")
                    st.write(f"**Training Date:** {model_info['trained_date']}")
                    st.write(f"**Training Samples:** {model_info['training_samples']:,}")
                    st.write(f"**Features:** {model_info['n_features']}")
                
                with col3:
                    st.markdown("**Actions**")
                    
                    col_actions1, col_actions2 = st.columns(2)
                    
                    with col_actions1:
                        if st.button(f"🔄 Retrain", key=f"retrain_{model_info['name']}"):
                            retrain_model(model_info['name'])
                    
                    with col_actions2:
                        if st.button(f"📊 Details", key=f"details_{model_info['name']}"):
                            show_model_details(model_info)
                
                # Feature importance
                st.markdown("**Feature Importance**")
                feature_importance = model_info.get('feature_importance', {})
                if feature_importance:
                    fig = px.bar(
                        x=list(feature_importance.values()),
                        y=list(feature_importance.keys()),
                        orientation='h',
                        title=f"Feature Importance - {model_info['name']}"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Model Training & Retraining")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Training Configuration")
            
            model_type = st.selectbox(
                "Select Model Type",
                ["XGBoost", "LightGBM", "RandomForest", "LogisticRegression", "SVM", "GradientBoosting"]
            )
            
            training_data_source = st.selectbox(
                "Training Data Source",
                ["Latest Batch", "Synthetic Data", "Combined", "Custom Upload"]
            )
            
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
            
            hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=True)
            
            cross_validation = st.checkbox("Enable Cross-Validation", value=True)
            
            if cross_validation:
                cv_folds = st.slider("CV Folds", 3, 10, 5)
        
        with col2:
            st.markdown("##### Advanced Settings")
            
            feature_selection = st.checkbox("Automatic Feature Selection", value=True)
            
            if feature_selection:
                feature_selection_method = st.selectbox(
                    "Feature Selection Method",
                    ["Recursive Feature Elimination", "SelectKBest", "L1 Regularization"]
                )
            
            early_stopping = st.checkbox("Early Stopping", value=True)
            
            if early_stopping:
                patience = st.slider("Patience (epochs)", 5, 50, 10)
            
            ensemble_method = st.selectbox(
                "Ensemble Method",
                ["None", "Voting", "Stacking", "Blending"]
            )
        
        if st.button("🚀 Start Training"):
            training_config = {
                'model_type': model_type,
                'data_source': training_data_source,
                'validation_split': validation_split,
                'hyperparameter_tuning': hyperparameter_tuning,
                'cross_validation': cross_validation,
                'feature_selection': feature_selection,
                'early_stopping': early_stopping,
                'ensemble_method': ensemble_method
            }
            
            start_model_training(training_config)
        
        # Training history
        st.markdown("##### Recent Training Jobs")
        
        training_history = get_training_history()
        st.dataframe(training_history, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("#### Model Version Control")
        
        # Model versions overview
        versions = version_manager.get_all_versions()
        
        st.markdown("##### Version History")
        
        for version in versions[:10]:  # Show last 10 versions
            with st.expander(f"Version {version['version']} - {version['model_name']} ({version['created_date']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Performance**")
                    st.metric("AUC", f"{version['performance']['auc']:.3f}")
                    st.metric("Accuracy", f"{version['performance']['accuracy']:.3f}")
                
                with col2:
                    st.markdown("**Metadata**")
                    st.write(f"**Status:** {version['status']}")
                    st.write(f"**Size:** {version['model_size']}")
                    st.write(f"**Training Time:** {version['training_time']}")
                
                with col3:
                    st.markdown("**Actions**")
                    
                    if version['status'] != 'active':
                        if st.button(f"✅ Deploy", key=f"deploy_{version['version']}"):
                            deploy_model_version(version['version'])
                    
                    if st.button(f"📥 Download", key=f"download_{version['version']}"):
                        download_model_version(version['version'])
                    
                    if version['status'] != 'active':
                        if st.button(f"🗑️ Delete", key=f"delete_{version['version']}"):
                            delete_model_version(version['version'])
        
        # Version comparison
        st.markdown("##### Version Comparison")
        
        selected_versions = st.multiselect(
            "Select versions to compare",
            [f"v{v['version']} - {v['model_name']}" for v in versions[:5]],
            max_selections=3
        )
        
        if len(selected_versions) >= 2:
            comparison_fig = create_version_comparison_chart(selected_versions, versions)
            st.plotly_chart(comparison_fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### A/B Testing")
        
        st.info("Configure and manage A/B tests between different model versions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Create New A/B Test")
            
            test_name = st.text_input("Test Name", "Model Comparison Test")
            
            model_a = st.selectbox("Model A", ["XGBoost v2.1", "LightGBM v1.8", "RandomForest v1.5"])
            model_b = st.selectbox("Model B", ["XGBoost v2.2", "LightGBM v1.9", "RandomForest v1.6"])
            
            traffic_split = st.slider("Traffic Split (% to Model A)", 10, 90, 50)
            
            test_duration = st.selectbox("Test Duration", ["1 day", "3 days", "1 week", "2 weeks", "1 month"])
            
            success_metric = st.selectbox("Success Metric", ["AUC Score", "Precision", "Recall", "F1 Score"])
            
            if st.button("🧪 Start A/B Test"):
                create_ab_test(test_name, model_a, model_b, traffic_split, test_duration, success_metric)
        
        with col2:
            st.markdown("##### Active A/B Tests")
            
            active_tests = get_active_ab_tests()
            
            for test in active_tests:
                with st.expander(f"🧪 {test['name']} (Running)"):
                    st.write(f"**Models:** {test['model_a']} vs {test['model_b']}")
                    st.write(f"**Split:** {test['traffic_split']}% / {100-test['traffic_split']}%")
                    st.write(f"**Started:** {test['start_date']}")
                    st.write(f"**Metric:** {test['success_metric']}")
                    
                    # Test results so far
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(f"{test['model_a']}", f"{test['results_a']['value']:.3f}", f"{test['results_a']['change']}")
                    
                    with col_b:
                        st.metric(f"{test['model_b']}", f"{test['results_b']['value']:.3f}", f"{test['results_b']['change']}")
                    
                    if st.button(f"⏹️ Stop Test", key=f"stop_{test['id']}"):
                        stop_ab_test(test['id'])

def show_advanced_data_generation():
    """Show advanced synthetic data generation for admins"""
    st.markdown("### 🎲 Advanced Data Generation")
    
    st.info("Unlimited synthetic data generation with advanced configuration options")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Generator Config", "📊 Batch Generation", "🎯 Scenario Testing"])
    
    with tab1:
        st.markdown("#### Advanced Generator Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Data Distribution Settings")
            
            num_records = st.number_input("Number of Records", 1000, 1000000, 10000)
            
            age_distribution = st.selectbox("Age Distribution", 
                                          ["Normal", "Uniform", "Beta", "Exponential"])
            
            if age_distribution == "Normal":
                age_mean = st.slider("Age Mean", 20, 70, 40)
                age_std = st.slider("Age Std Dev", 5, 20, 12)
            
            income_distribution = st.selectbox("Income Distribution",
                                             ["Log-Normal", "Pareto", "Normal", "Custom"])
            
            if income_distribution == "Log-Normal":
                income_mean = st.number_input("Income Mean (log)", 8.0, 12.0, 11.0)
                income_std = st.slider("Income Std (log)", 0.1, 2.0, 0.5)
            
            credit_score_model = st.selectbox("Credit Score Model",
                                            ["FICO", "VantageScore", "Custom", "Mixed"])
        
        with col2:
            st.markdown("##### Economic Scenario Settings")
            
            economic_scenario = st.selectbox("Economic Scenario",
                                           ["Normal", "Recession", "Boom", "High Inflation", "Custom"])
            
            default_rate_target = st.slider("Target Default Rate (%)", 1.0, 30.0, 8.0)
            
            correlation_strength = st.slider("Feature Correlation Strength", 0.0, 1.0, 0.3)
            
            noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1)
            
            seasonal_effects = st.checkbox("Include Seasonal Effects", value=True)
            
            if seasonal_effects:
                seasonality_strength = st.slider("Seasonality Strength", 0.1, 1.0, 0.2)
        
        # Advanced feature configuration
        st.markdown("##### Feature Engineering Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_derived_features = st.checkbox("Derived Features", value=True)
            include_interaction_terms = st.checkbox("Interaction Terms", value=True)
        
        with col2:
            include_categorical_features = st.checkbox("Categorical Features", value=True)
            include_time_features = st.checkbox("Time-based Features", value=True)
        
        with col3:
            include_external_factors = st.checkbox("External Economic Factors", value=False)
            include_behavioral_features = st.checkbox("Behavioral Features", value=True)
        
        if st.button("🎲 Generate Advanced Dataset"):
            config = {
                'num_records': num_records,
                'age_distribution': age_distribution,
                'income_distribution': income_distribution,
                'credit_score_model': credit_score_model,
                'economic_scenario': economic_scenario,
                'default_rate_target': default_rate_target,
                'correlation_strength': correlation_strength,
                'noise_level': noise_level,
                'seasonal_effects': seasonal_effects,
                'derived_features': include_derived_features,
                'interaction_terms': include_interaction_terms,
                'categorical_features': include_categorical_features,
                'time_features': include_time_features,
                'external_factors': include_external_factors,
                'behavioral_features': include_behavioral_features
            }
            
            generate_advanced_synthetic_data(config)
    
    with tab2:
        st.markdown("#### Batch Data Generation")
        
        st.markdown("Generate multiple datasets for comprehensive testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_configs = st.number_input("Number of Datasets", 1, 20, 5)
            records_per_batch = st.number_input("Records per Dataset", 1000, 100000, 10000)
            
            scenarios = st.multiselect(
                "Economic Scenarios",
                ["Normal", "Recession", "Boom", "High Inflation", "Market Crash"],
                default=["Normal", "Recession"]
            )
        
        with col2:
            export_format = st.selectbox("Export Format", ["CSV", "Parquet", "JSON", "SQL"])
            
            include_metadata = st.checkbox("Include Metadata", value=True)
            
            compress_files = st.checkbox("Compress Files", value=True)
        
        if st.button("🏭 Generate Batch Datasets"):
            generate_batch_synthetic_data(batch_configs, records_per_batch, scenarios, 
                                        export_format, include_metadata, compress_files)
    
    with tab3:
        st.markdown("#### Scenario Testing Data")
        
        st.markdown("Generate data for specific testing scenarios")
        
        scenario_type = st.selectbox(
            "Scenario Type",
            ["Edge Cases", "Stress Testing", "Model Drift", "Bias Testing", "Performance Testing"]
        )
        
        if scenario_type == "Edge Cases":
            st.markdown("##### Edge Case Configuration")
            
            include_extreme_values = st.checkbox("Extreme Values", value=True)
            include_missing_data = st.checkbox("Missing Data Patterns", value=True)
            include_outliers = st.checkbox("Statistical Outliers", value=True)
            
            outlier_percentage = st.slider("Outlier Percentage", 1, 20, 5)
            
        elif scenario_type == "Stress Testing":
            st.markdown("##### Stress Testing Configuration")
            
            high_volume = st.checkbox("High Volume Scenarios", value=True)
            rapid_changes = st.checkbox("Rapid Economic Changes", value=True)
            extreme_correlations = st.checkbox("Extreme Feature Correlations", value=True)
            
        elif scenario_type == "Bias Testing":
            st.markdown("##### Bias Testing Configuration")
            
            protected_attributes = st.multiselect(
                "Protected Attributes to Test",
                ["Age", "Gender", "Race", "Geographic Location"],
                default=["Age"]
            )
            
            bias_scenarios = st.multiselect(
                "Bias Scenarios",
                ["Historical Bias", "Representation Bias", "Algorithmic Bias"],
                default=["Historical Bias"]
            )
        
        if st.button("🎯 Generate Scenario Data"):
            generate_scenario_testing_data(scenario_type)

def show_stress_testing():
    """Show stress testing interface"""
    st.markdown("### 🏋️ Stress Testing")
    
    st.info("Test model performance under extreme conditions and edge cases")
    
    tab1, tab2, tab3 = st.tabs(["⚡ Load Testing", "🌊 Data Drift", "🎯 Edge Cases"])
    
    with tab1:
        st.markdown("#### Load Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Test Configuration")
            
            concurrent_users = st.slider("Concurrent Users", 1, 1000, 50)
            requests_per_second = st.slider("Requests per Second", 1, 100, 10)
            test_duration = st.selectbox("Test Duration", ["1 minute", "5 minutes", "15 minutes", "1 hour"])
            
            request_type = st.selectbox("Request Type", ["Prediction", "Batch Processing", "Mixed"])
        
        with col2:
            st.markdown("##### Expected Outcomes")
            
            st.metric("Expected Response Time", "< 200ms")
            st.metric("Expected Success Rate", "> 99%")
            st.metric("Expected Throughput", f"{requests_per_second * concurrent_users} req/s")
        
        if st.button("🚀 Start Load Test"):
            run_load_test(concurrent_users, requests_per_second, test_duration, request_type)
        
        # Recent load test results
        st.markdown("##### Recent Load Test Results")
        
        load_test_results = get_load_test_results()
        st.dataframe(load_test_results, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### Data Drift Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Drift Simulation")
            
            drift_type = st.selectbox("Drift Type", 
                                    ["Gradual", "Sudden", "Seasonal", "Cyclical"])
            
            affected_features = st.multiselect(
                "Affected Features",
                ["credit_score", "income", "age", "debt_ratio", "employment_length"],
                default=["credit_score", "income"]
            )
            
            drift_magnitude = st.slider("Drift Magnitude", 0.1, 2.0, 0.5)
            
            simulation_periods = st.slider("Simulation Periods", 1, 12, 6)
        
        with col2:
            st.markdown("##### Drift Detection Settings")
            
            detection_method = st.selectbox("Detection Method", 
                                          ["KS Test", "PSI", "Wasserstein Distance", "All Methods"])
            
            alert_threshold = st.slider("Alert Threshold", 0.01, 0.1, 0.05)
            
            monitoring_window = st.selectbox("Monitoring Window", 
                                           ["Daily", "Weekly", "Monthly"])
        
        if st.button("🌊 Simulate Data Drift"):
            simulate_data_drift(drift_type, affected_features, drift_magnitude, 
                              simulation_periods, detection_method, alert_threshold)
        
        # Drift detection results
        st.markdown("##### Drift Detection Results")
        
        if 'drift_results' in st.session_state:
            drift_results = st.session_state.drift_results
            
            fig = px.line(drift_results, x='period', y='drift_score', color='feature',
                         title="Feature Drift Over Time")
            fig.add_hline(y=alert_threshold, line_dash="dash", line_color="red",
                         annotation_text="Alert Threshold")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Edge Case Testing")
        
        edge_case_categories = st.multiselect(
            "Select Edge Case Categories",
            ["Extreme Values", "Missing Data", "Invalid Inputs", "Boundary Conditions", "Data Quality Issues"],
            default=["Extreme Values", "Missing Data"]
        )
        
        for category in edge_case_categories:
            with st.expander(f"📋 {category} Test Configuration"):
                if category == "Extreme Values":
                    test_extreme_low = st.checkbox(f"Test Extremely Low Values - {category}", value=True)
                    test_extreme_high = st.checkbox(f"Test Extremely High Values - {category}", value=True)
                    
                elif category == "Missing Data":
                    missing_data_patterns = st.multiselect(
                        "Missing Data Patterns",
                        ["Random Missing", "Systematic Missing", "Complete Feature Missing"],
                        default=["Random Missing"]
                    )
                    missing_percentage = st.slider(f"Missing Percentage - {category}", 5, 50, 20)
                
                elif category == "Boundary Conditions":
                    test_min_boundaries = st.checkbox(f"Test Minimum Boundaries - {category}", value=True)
                    test_max_boundaries = st.checkbox(f"Test Maximum Boundaries - {category}", value=True)
        
        if st.button("🎯 Run Edge Case Tests"):
            run_edge_case_tests(edge_case_categories)
        
        # Edge case test results
        if 'edge_case_results' in st.session_state:
            st.markdown("##### Edge Case Test Results")
            
            results = st.session_state.edge_case_results
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tests Passed", results['passed'], f"↗ {results['passed_change']}")
            
            with col2:
                st.metric("Tests Failed", results['failed'], f"↘ {results['failed_change']}")
            
            with col3:
                st.metric("Success Rate", f"{results['success_rate']:.1%}")
            
            # Detailed results
            st.dataframe(results['detailed_results'], use_container_width=True, hide_index=True)

def show_schema_editor():
    """Show schema editing interface"""
    st.markdown("### 📋 Schema Editor")
    
    st.info("Define and modify data schemas for different loan types and use cases")
    
    tab1, tab2, tab3 = st.tabs(["📝 Current Schema", "✏️ Schema Editor", "📁 Schema Templates"])
    
    with tab1:
        st.markdown("#### Current Data Schema")
        
        current_schema = get_current_schema()
        
        st.markdown("##### Required Fields")
        required_df = pd.DataFrame([
            field for field in current_schema['fields'] 
            if field.get('required', False)
        ])
        st.dataframe(required_df, use_container_width=True, hide_index=True)
        
        st.markdown("##### Optional Fields")
        optional_df = pd.DataFrame([
            field for field in current_schema['fields'] 
            if not field.get('required', False)
        ])
        st.dataframe(optional_df, use_container_width=True, hide_index=True)
        
        st.markdown("##### Data Validation Rules")
        st.json(current_schema['validation_rules'])
    
    with tab2:
        st.markdown("#### Schema Editor")
        
        schema_name = st.text_input("Schema Name", "custom_loan_schema")
        schema_description = st.text_area("Schema Description", "Custom loan application schema")
        
        st.markdown("##### Add/Edit Fields")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            field_name = st.text_input("Field Name")
        
        with col2:
            field_type = st.selectbox("Field Type", 
                                    ["string", "integer", "float", "boolean", "date", "categorical"])
        
        with col3:
            is_required = st.checkbox("Required")
        
        with col4:
            field_description = st.text_input("Description")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if field_type in ["integer", "float"]:
                min_value = st.number_input("Minimum Value", value=0)
                max_value = st.number_input("Maximum Value", value=1000000)
            
            elif field_type == "string":
                max_length = st.number_input("Maximum Length", 1, 1000, 100)
                pattern = st.text_input("Validation Pattern (regex)")
            
            elif field_type == "categorical":
                allowed_values = st.text_area("Allowed Values (one per line)")
        
        with col2:
            default_value = st.text_input("Default Value")
            transformation = st.selectbox("Data Transformation", 
                                        ["none", "lowercase", "uppercase", "normalize", "standardize"])
        
        if st.button("➕ Add Field"):
            add_field_to_schema(field_name, field_type, is_required, field_description,
                              min_value if field_type in ["integer", "float"] else None,
                              max_value if field_type in ["integer", "float"] else None,
                              max_length if field_type == "string" else None,
                              pattern if field_type == "string" else None,
                              allowed_values.split('\n') if field_type == "categorical" else None,
                              default_value, transformation)
        
        # Current schema fields editor
        st.markdown("##### Current Schema Fields")
        
        if 'custom_schema_fields' not in st.session_state:
            st.session_state.custom_schema_fields = []
        
        for i, field in enumerate(st.session_state.custom_schema_fields):
            with st.expander(f"📝 {field['name']} ({field['type']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Type:** {field['type']}")
                    st.write(f"**Required:** {'Yes' if field['required'] else 'No'}")
                
                with col2:
                    st.write(f"**Description:** {field['description']}")
                    if field.get('default_value'):
                        st.write(f"**Default:** {field['default_value']}")
                
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_field_{i}"):
                        st.session_state.custom_schema_fields.pop(i)
                        st.rerun()
        
        if st.button("💾 Save Schema"):
            save_custom_schema(schema_name, schema_description, st.session_state.custom_schema_fields)
    
    with tab3:
        st.markdown("#### Schema Templates")
        
        templates = get_schema_templates()
        
        template_choice = st.selectbox("Select Template", 
                                     ["Custom", "Personal Loan", "Mortgage", "Credit Card", "Auto Loan", "Business Loan"])
        
        if template_choice != "Custom":
            template = templates.get(template_choice.lower().replace(' ', '_'))
            
            if template:
                st.markdown(f"##### {template_choice} Schema Template")
                
                st.markdown("**Description:**")
                st.write(template['description'])
                
                st.markdown("**Fields:**")
                template_df = pd.DataFrame(template['fields'])
                st.dataframe(template_df, use_container_width=True, hide_index=True)
                
                if st.button(f"📥 Load {template_choice} Template"):
                    load_schema_template(template)
        
        # Import/Export schemas
        st.markdown("---")
        st.markdown("##### Import/Export Schemas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Import Schema**")
            uploaded_schema = st.file_uploader("Upload Schema JSON", type=['json'])
            
            if uploaded_schema and st.button("📥 Import Schema"):
                import_schema_file(uploaded_schema)
        
        with col2:
            st.markdown("**Export Schema**")
            
            if st.button("📤 Export Current Schema"):
                export_current_schema()

def show_user_management():
    """Show user management interface"""
    st.markdown("### 👥 User Management")
    
    tab1, tab2, tab3 = st.tabs(["👤 User List", "📊 User Analytics", "⚙️ User Settings"])
    
    with tab1:
        st.markdown("#### Registered Users")
        
        users = get_all_users()
        
        # User statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users))
        
        with col2:
            admin_count = len([u for u in users if u['role'] == 'admin'])
            st.metric("Admin Users", admin_count)
        
        with col3:
            active_count = len([u for u in users if u.get('status') == 'active'])
            st.metric("Active Users", active_count)
        
        with col4:
            today = datetime.now().date()
            new_today = len([u for u in users if u.get('created_date', '').startswith(str(today))])
            st.metric("New Today", new_today)
        
        # User list with actions
        st.markdown("##### User Directory")
        
        for user in users:
            with st.expander(f"👤 {user['username']} ({user['role']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Username:** {user['username']}")
                    st.write(f"**Role:** {user['role']}")
                    st.write(f"**Status:** {user.get('status', 'active')}")
                
                with col2:
                    st.write(f"**Created:** {user.get('created_at', 'N/A')}")
                    st.write(f"**Last Login:** {user.get('last_login', 'Never')}")
                    st.write(f"**Predictions:** {user.get('prediction_count', 0)}")
                
                with col3:
                    st.markdown("**Actions**")
                    
                    col_actions1, col_actions2, col_actions3 = st.columns(3)
                    
                    with col_actions1:
                        if user['username'] != st.session_state.username:  # Don't allow self-action
                            if st.button(f"🔒 {'Deactivate' if user.get('status') == 'active' else 'Activate'}", 
                                       key=f"toggle_{user['username']}"):
                                toggle_user_status(user['username'])
                    
                    with col_actions2:
                        if st.button(f"🔄 Reset Pwd", key=f"reset_{user['username']}"):
                            reset_user_password(user['username'])
                    
                    with col_actions3:
                        if user['username'] != st.session_state.username:
                            if st.button(f"🗑️ Delete", key=f"delete_{user['username']}"):
                                delete_user(user['username'])
    
    with tab2:
        st.markdown("#### User Analytics")
        
        # User activity over time
        user_activity = get_user_activity_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Login Activity")
            
            login_fig = px.line(user_activity, x='date', y='logins', 
                              title="Daily Login Activity")
            st.plotly_chart(login_fig, use_container_width=True)
        
        with col2:
            st.markdown("##### User Registration Trends")
            
            registration_fig = px.bar(user_activity, x='date', y='new_registrations',
                                    title="New User Registrations")
            st.plotly_chart(registration_fig, use_container_width=True)
        
        # Usage patterns
        st.markdown("##### Usage Patterns")
        
        usage_data = get_user_usage_patterns()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Feature Usage**")
            feature_usage_fig = px.pie(usage_data, values='usage_count', names='feature',
                                     title="Feature Usage Distribution")
            st.plotly_chart(feature_usage_fig, use_container_width=True)
        
        with col2:
            st.markdown("**User Engagement**")
            engagement_fig = px.histogram(usage_data, x='session_duration',
                                        title="Session Duration Distribution")
            st.plotly_chart(engagement_fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### User Settings & Permissions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Global User Settings")
            
            default_role = st.selectbox("Default Role for New Users", ["client", "admin"])
            
            require_email_verification = st.checkbox("Require Email Verification", value=False)
            
            password_min_length = st.slider("Minimum Password Length", 6, 20, 8)
            
            session_timeout = st.selectbox("Session Timeout", 
                                         ["15 minutes", "30 minutes", "1 hour", "2 hours", "Never"])
            
            max_login_attempts = st.slider("Max Login Attempts", 3, 10, 5)
        
        with col2:
            st.markdown("##### Permission Settings")
            
            client_permissions = st.multiselect(
                "Client Permissions",
                ["Upload Data", "Generate Synthetic Data", "View Predictions", 
                 "Export Results", "What-If Analysis", "View Model Insights"],
                default=["Upload Data", "View Predictions", "What-If Analysis"]
            )
            
            admin_permissions = st.multiselect(
                "Admin Permissions",
                ["Model Management", "User Management", "System Logs", 
                 "Schema Editing", "Stress Testing", "A/B Testing"],
                default=["Model Management", "User Management", "System Logs", 
                        "Schema Editing", "Stress Testing", "A/B Testing"]
            )
        
        if st.button("💾 Save Settings"):
            save_user_settings({
                'default_role': default_role,
                'require_email_verification': require_email_verification,
                'password_min_length': password_min_length,
                'session_timeout': session_timeout,
                'max_login_attempts': max_login_attempts,
                'client_permissions': client_permissions,
                'admin_permissions': admin_permissions
            })

def show_system_logs():
    """Show system logs and monitoring"""
    st.markdown("### 📊 System Logs & Monitoring")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Activity Logs", "⚠️ Error Logs", "📈 Performance", "🔒 Security"])
    
    with tab1:
        st.markdown("#### System Activity Logs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR", "DEBUG"])
        
        with col2:
            date_range = st.selectbox("Date Range", ["Today", "Last 7 days", "Last 30 days", "Custom"])
        
        with col3:
            component = st.selectbox("Component", ["All", "Authentication", "ML Pipeline", "API", "Database"])
        
        # Get filtered logs
        activity_logs = get_activity_logs(log_level, date_range, component)
        
        st.markdown("##### Recent Activity")
        st.dataframe(activity_logs, use_container_width=True, hide_index=True)
        
        # Download logs
        if st.button("📥 Download Logs"):
            download_logs(activity_logs, "activity_logs")
    
    with tab2:
        st.markdown("#### Error Logs & Debugging")
        
        error_logs = get_error_logs()
        
        # Error summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_errors = len(error_logs)
            st.metric("Total Errors (24h)", total_errors, "↓ 12")
        
        with col2:
            critical_errors = len([e for e in error_logs if e['severity'] == 'CRITICAL'])
            st.metric("Critical Errors", critical_errors, "→ 0")
        
        with col3:
            resolved_errors = len([e for e in error_logs if e.get('status') == 'resolved'])
            st.metric("Resolved", resolved_errors)
        
        with col4:
            error_rate = (total_errors / 1000) * 100 if total_errors > 0 else 0
            st.metric("Error Rate", f"{error_rate:.2f}%", "↓ 0.3%")
        
        # Error trends
        st.markdown("##### Error Trends")
        
        error_trends = get_error_trends()
        error_fig = px.line(error_trends, x='hour', y='error_count', color='severity',
                           title="Hourly Error Trends")
        st.plotly_chart(error_fig, use_container_width=True)
        
        # Detailed error logs
        st.markdown("##### Detailed Error Logs")
        
        for error in error_logs[:20]:  # Show last 20 errors
            with st.expander(f"❌ {error['timestamp']} - {error['error_type']} ({error['severity']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Component:** {error['component']}")
                    st.write(f"**User:** {error.get('user', 'System')}")
                    st.write(f"**Status:** {error.get('status', 'Open')}")
                
                with col2:
                    st.write(f"**Error Message:** {error['message']}")
                    if error.get('stack_trace'):
                        st.code(error['stack_trace'], language='python')
                
                if error.get('status') != 'resolved':
                    if st.button(f"✅ Mark Resolved", key=f"resolve_{error['id']}"):
                        mark_error_resolved(error['id'])
    
    with tab3:
        st.markdown("#### Performance Monitoring")
        
        # System performance metrics
        performance_metrics = get_performance_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Response Time", f"{performance_metrics['avg_response_time']:.0f}ms", 
                     f"↓ {performance_metrics['response_time_change']:.0f}ms")
        
        with col2:
            st.metric("Requests/Hour", f"{performance_metrics['requests_per_hour']:,}", 
                     f"↗ {performance_metrics['requests_change']:,}")
        
        with col3:
            st.metric("Success Rate", f"{performance_metrics['success_rate']:.1%}", 
                     f"↗ {performance_metrics['success_rate_change']:.1%}")
        
        with col4:
            st.metric("CPU Usage", f"{performance_metrics['cpu_usage']:.1f}%", 
                     f"↘ {performance_metrics['cpu_change']:.1f}%")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Response Time Trends")
            
            response_time_data = get_response_time_data()
            response_fig = px.line(response_time_data, x='timestamp', y='response_time',
                                 title="Response Time Over Time")
            st.plotly_chart(response_fig, use_container_width=True)
        
        with col2:
            st.markdown("##### System Resource Usage")
            
            resource_data = get_resource_usage_data()
            resource_fig = px.line(resource_data, x='timestamp', 
                                 y=['cpu_usage', 'memory_usage', 'disk_usage'],
                                 title="Resource Usage Over Time")
            st.plotly_chart(resource_fig, use_container_width=True)
        
        # Database performance
        st.markdown("##### Database Performance")
        
        db_metrics = get_database_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Query Time (avg)", f"{db_metrics['avg_query_time']:.2f}ms")
        
        with col2:
            st.metric("Connections", db_metrics['active_connections'])
        
        with col3:
            st.metric("Cache Hit Rate", f"{db_metrics['cache_hit_rate']:.1%}")
    
    with tab4:
        st.markdown("#### Security Monitoring")
        
        security_events = get_security_events()
        
        # Security summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            failed_logins = len([e for e in security_events if e['type'] == 'failed_login'])
            st.metric("Failed Logins (24h)", failed_logins, "↓ 3")
        
        with col2:
            suspicious_activity = len([e for e in security_events if e['severity'] == 'HIGH'])
            st.metric("Suspicious Activity", suspicious_activity, "→ 0")
        
        with col3:
            blocked_ips = len(set([e['ip'] for e in security_events if e.get('blocked')]))
            st.metric("Blocked IPs", blocked_ips)
        
        with col4:
            security_score = calculate_security_score()
            st.metric("Security Score", f"{security_score:.1f}/10", "→ 0")
        
        # Security events
        st.markdown("##### Recent Security Events")
        
        for event in security_events[:15]:
            severity_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}
            severity_icon = severity_color.get(event['severity'], "⚪")
            
            with st.expander(f"{severity_icon} {event['timestamp']} - {event['type']} ({event['severity']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**IP Address:** {event['ip']}")
                    st.write(f"**User Agent:** {event.get('user_agent', 'N/A')}")
                    st.write(f"**Location:** {event.get('location', 'Unknown')}")
                
                with col2:
                    st.write(f"**Details:** {event['details']}")
                    st.write(f"**Action Taken:** {event.get('action_taken', 'None')}")
                
                if event['severity'] in ['HIGH', 'CRITICAL'] and not event.get('resolved'):
                    col_action1, col_action2 = st.columns(2)
                    
                    with col_action1:
                        if st.button(f"🚫 Block IP", key=f"block_{event['id']}"):
                            block_ip_address(event['ip'])
                    
                    with col_action2:
                        if st.button(f"✅ Mark Safe", key=f"safe_{event['id']}"):
                            mark_security_event_safe(event['id'])

# Helper functions for admin functionality
def retrain_all_models():
    """Retrain all ML models"""
    with st.spinner("Retraining all models..."):
        import time
        time.sleep(3)  # Simulate training time
        st.success("All models retrained successfully!")

def clean_system_cache():
    """Clean system cache"""
    with st.spinner("Cleaning system cache..."):
        import time
        time.sleep(1)
        st.success("System cache cleaned!")

def generate_system_report():
    """Generate comprehensive system report"""
    with st.spinner("Generating system report..."):
        import time
        time.sleep(2)
        st.success("System report generated! Check your downloads.")

def retrain_model(model_name):
    """Retrain specific model"""
    with st.spinner(f"Retraining {model_name}..."):
        import time
        time.sleep(2)
        st.success(f"{model_name} retrained successfully!")

def show_model_details(model_info):
    """Show detailed model information"""
    st.session_state.show_model_details = model_info
    st.info(f"Detailed information for {model_info['name']} loaded")

def start_model_training(config):
    """Start model training with given configuration"""
    with st.spinner("Starting model training..."):
        import time
        time.sleep(3)
        st.success(f"Training started for {config['model_type']}!")

def get_training_history():
    """Get model training history"""
    return pd.DataFrame({
        'Timestamp': ['2024-09-04 10:30', '2024-09-03 15:45', '2024-09-02 09:15'],
        'Model': ['XGBoost', 'LightGBM', 'RandomForest'],
        'Status': ['✅ Completed', '✅ Completed', '❌ Failed'],
        'Duration': ['12m 34s', '8m 12s', '5m 45s'],
        'AUC Score': [0.847, 0.832, None]
    })

def deploy_model_version(version):
    """Deploy specific model version"""
    with st.spinner(f"Deploying model version {version}..."):
        import time
        time.sleep(2)
        st.success(f"Model version {version} deployed successfully!")

def download_model_version(version):
    """Download model version"""
    st.info(f"Downloading model version {version}...")

def delete_model_version(version):
    """Delete model version"""
    st.warning(f"Model version {version} deleted!")

def create_version_comparison_chart(selected_versions, versions):
    """Create version comparison chart"""
    # Sample comparison data
    comparison_data = pd.DataFrame({
        'Version': ['v2.1', 'v2.2', 'v2.3'],
        'AUC': [0.847, 0.852, 0.849],
        'Precision': [0.823, 0.835, 0.827],
        'Recall': [0.756, 0.748, 0.762]
    })
    
    fig = px.line(comparison_data.melt(id_vars=['Version'], var_name='Metric', value_name='Score'),
                  x='Version', y='Score', color='Metric', title="Model Version Comparison")
    return fig

def create_ab_test(test_name, model_a, model_b, traffic_split, test_duration, success_metric):
    """Create new A/B test"""
    with st.spinner("Creating A/B test..."):
        import time
        time.sleep(2)
        st.success(f"A/B test '{test_name}' created successfully!")

def get_active_ab_tests():
    """Get currently active A/B tests"""
    return [
        {
            'id': 'test_001',
            'name': 'XGBoost vs LightGBM',
            'model_a': 'XGBoost v2.1',
            'model_b': 'LightGBM v1.9',
            'traffic_split': 50,
            'start_date': '2024-09-01',
            'success_metric': 'AUC Score',
            'results_a': {'value': 0.847, 'change': '+0.003'},
            'results_b': {'value': 0.852, 'change': '+0.008'}
        }
    ]

def stop_ab_test(test_id):
    """Stop A/B test"""
    st.success(f"A/B test {test_id} stopped!")

def generate_advanced_synthetic_data(config):
    """Generate advanced synthetic data"""
    with st.spinner("Generating advanced synthetic data..."):
        import time
        time.sleep(3)
        st.success(f"Generated {config['num_records']:,} records with advanced configuration!")

def generate_batch_synthetic_data(batch_configs, records_per_batch, scenarios, export_format, include_metadata, compress_files):
    """Generate batch synthetic data"""
    with st.spinner("Generating batch datasets..."):
        import time
        time.sleep(5)
        st.success(f"Generated {batch_configs} datasets with {records_per_batch:,} records each!")

def generate_scenario_testing_data(scenario_type):
    """Generate scenario testing data"""
    with st.spinner(f"Generating {scenario_type} testing data..."):
        import time
        time.sleep(3)
        st.success(f"{scenario_type} testing data generated!")

def run_load_test(concurrent_users, requests_per_second, test_duration, request_type):
    """Run load test"""
    with st.spinner("Running load test..."):
        import time
        time.sleep(10)
        st.success("Load test completed successfully!")

def get_load_test_results():
    """Get load test results"""
    return pd.DataFrame({
        'Test Date': ['2024-09-04', '2024-09-03', '2024-09-02'],
        'Users': [50, 30, 100],
        'RPS': [10, 15, 5],
        'Avg Response': ['145ms', '167ms', '234ms'],
        'Success Rate': ['99.8%', '99.5%', '98.9%']
    })

def simulate_data_drift(drift_type, affected_features, drift_magnitude, simulation_periods, detection_method, alert_threshold):
    """Simulate data drift"""
    with st.spinner("Simulating data drift..."):
        import time
        time.sleep(3)
        
        # Generate sample drift results
        drift_data = []
        for period in range(1, simulation_periods + 1):
            for feature in affected_features:
                drift_score = drift_magnitude * (period / simulation_periods) + np.random.normal(0, 0.05)
                drift_data.append({
                    'period': period,
                    'feature': feature,
                    'drift_score': drift_score
                })
        
        st.session_state.drift_results = pd.DataFrame(drift_data)
        st.success("Data drift simulation completed!")

def run_edge_case_tests(edge_case_categories):
    """Run edge case tests"""
    with st.spinner("Running edge case tests..."):
        import time
        time.sleep(4)
        
        # Generate sample results
        results = {
            'passed': 87,
            'failed': 13,
            'passed_change': 5,
            'failed_change': -2,
            'success_rate': 0.87,
            'detailed_results': pd.DataFrame({
                'Test Case': ['Extreme Low Income', 'Missing Credit Score', 'Zero Debt Ratio', 'Invalid Age'],
                'Category': ['Extreme Values', 'Missing Data', 'Boundary Conditions', 'Invalid Inputs'],
                'Status': ['✅ Passed', '❌ Failed', '✅ Passed', '❌ Failed'],
                'Error Rate': ['0.0%', '12.3%', '0.0%', '8.7%']
            })
        }
        
        st.session_state.edge_case_results = results
        st.success("Edge case tests completed!")

def get_current_schema():
    """Get current data schema"""
    return {
        'name': 'default_loan_schema',
        'version': '1.0',
        'fields': [
            {'name': 'age', 'type': 'integer', 'required': True, 'description': 'Applicant age'},
            {'name': 'annual_income', 'type': 'float', 'required': True, 'description': 'Annual income'},
            {'name': 'credit_score', 'type': 'integer', 'required': True, 'description': 'Credit score'},
            {'name': 'debt_to_income_ratio', 'type': 'float', 'required': True, 'description': 'Debt to income ratio'},
            {'name': 'employment_length', 'type': 'categorical', 'required': False, 'description': 'Employment length'},
        ],
        'validation_rules': {
            'age': {'min': 18, 'max': 100},
            'annual_income': {'min': 0, 'max': 10000000},
            'credit_score': {'min': 300, 'max': 850}
        }
    }

def add_field_to_schema(field_name, field_type, is_required, field_description, min_value, max_value, max_length, pattern, allowed_values, default_value, transformation):
    """Add field to custom schema"""
    if 'custom_schema_fields' not in st.session_state:
        st.session_state.custom_schema_fields = []
    
    new_field = {
        'name': field_name,
        'type': field_type,
        'required': is_required,
        'description': field_description,
        'default_value': default_value,
        'transformation': transformation
    }
    
    if min_value is not None:
        new_field['min_value'] = min_value
    if max_value is not None:
        new_field['max_value'] = max_value
    if max_length:
        new_field['max_length'] = max_length
    if pattern:
        new_field['pattern'] = pattern
    if allowed_values:
        new_field['allowed_values'] = allowed_values
    
    st.session_state.custom_schema_fields.append(new_field)
    st.success(f"Field '{field_name}' added to schema!")

def save_custom_schema(schema_name, schema_description, fields):
    """Save custom schema"""
    st.success(f"Schema '{schema_name}' saved successfully!")

def get_schema_templates():
    """Get schema templates"""
    return {
        'personal_loan': {
            'description': 'Schema for personal loan applications',
            'fields': [
                {'name': 'age', 'type': 'integer', 'required': True},
                {'name': 'income', 'type': 'float', 'required': True},
                {'name': 'credit_score', 'type': 'integer', 'required': True},
                {'name': 'loan_amount', 'type': 'float', 'required': True},
            ]
        }
    }

def load_schema_template(template):
    """Load schema template"""
    st.session_state.custom_schema_fields = template['fields']
    st.success("Schema template loaded!")

def import_schema_file(uploaded_file):
    """Import schema from file"""
    st.success("Schema imported successfully!")

def export_current_schema():
    """Export current schema"""
    st.success("Schema exported successfully!")

def get_all_users():
    """Get all registered users"""
    return [
        {
            'username': 'admin',
            'role': 'admin',
            'status': 'active',
            'created_at': '2024-01-01',
            'last_login': '2024-09-04 12:00',
            'prediction_count': 1247
        },
        {
            'username': 'demo_client',
            'role': 'client',
            'status': 'active',
            'created_at': '2024-02-15',
            'last_login': '2024-09-03 15:30',
            'prediction_count': 89
        }
    ]

def toggle_user_status(username):
    """Toggle user active status"""
    st.success(f"User {username} status toggled!")

def reset_user_password(username):
    """Reset user password"""
    st.success(f"Password reset for {username}!")

def delete_user(username):
    """Delete user"""
    st.warning(f"User {username} deleted!")

def get_user_activity_data():
    """Get user activity data"""
    dates = pd.date_range(start='2024-08-01', end='2024-09-04', freq='D')
    return pd.DataFrame({
        'date': dates,
        'logins': np.random.poisson(25, len(dates)),
        'new_registrations': np.random.poisson(2, len(dates))
    })

def get_user_usage_patterns():
    """Get user usage patterns"""
    features = ['Predictions', 'Data Upload', 'Dashboard', 'What-If Analysis', 'Export']
    return pd.DataFrame({
        'feature': features,
        'usage_count': [1247, 456, 890, 234, 123],
        'session_duration': np.random.normal(15, 5, len(features))
    })

def save_user_settings(settings):
    """Save user settings"""
    st.success("User settings saved successfully!")

def get_activity_logs(log_level, date_range, component):
    """Get activity logs"""
    return pd.DataFrame({
        'Timestamp': ['2024-09-04 12:05', '2024-09-04 11:45', '2024-09-04 11:30'],
        'Level': ['INFO', 'INFO', 'WARNING'],
        'Component': ['Authentication', 'ML Pipeline', 'API'],
        'User': ['admin', 'demo_client', 'system'],
        'Message': ['User logged in successfully', 'Model prediction completed', 'API rate limit approached'],
        'IP': ['192.168.1.100', '192.168.1.101', '192.168.1.102']
    })

def download_logs(logs, filename):
    """Download logs"""
    st.success(f"Logs downloaded as {filename}.csv")

def get_error_logs():
    """Get error logs"""
    return [
        {
            'id': 'err_001',
            'timestamp': '2024-09-04 11:30',
            'error_type': 'ValidationError',
            'severity': 'MEDIUM',
            'component': 'Data Processing',
            'message': 'Invalid data format in uploaded file',
            'user': 'demo_client',
            'status': 'open'
        }
    ]

def get_error_trends():
    """Get error trends"""
    hours = list(range(24))
    return pd.DataFrame({
        'hour': hours,
        'error_count': np.random.poisson(2, 24),
        'severity': ['LOW'] * 24
    })

def mark_error_resolved(error_id):
    """Mark error as resolved"""
    st.success(f"Error {error_id} marked as resolved!")

def get_performance_metrics():
    """Get performance metrics"""
    return {
        'avg_response_time': 142,
        'response_time_change': -15,
        'requests_per_hour': 1247,
        'requests_change': 123,
        'success_rate': 0.998,
        'success_rate_change': 0.002,
        'cpu_usage': 45.2,
        'cpu_change': -3.1
    }

def get_response_time_data():
    """Get response time data"""
    timestamps = pd.date_range(start='2024-09-04', periods=24, freq='H')
    return pd.DataFrame({
        'timestamp': timestamps,
        'response_time': np.random.normal(150, 20, 24)
    })

def get_resource_usage_data():
    """Get resource usage data"""
    timestamps = pd.date_range(start='2024-09-04', periods=24, freq='H')
    return pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.random.normal(45, 10, 24),
        'memory_usage': np.random.normal(60, 15, 24),
        'disk_usage': np.random.normal(30, 5, 24)
    })

def get_database_metrics():
    """Get database metrics"""
    return {
        'avg_query_time': 12.34,
        'active_connections': 23,
        'cache_hit_rate': 0.87
    }

def get_security_events():
    """Get security events"""
    return [
        {
            'id': 'sec_001',
            'timestamp': '2024-09-04 11:45',
            'type': 'failed_login',
            'severity': 'MEDIUM',
            'ip': '192.168.1.105',
            'details': 'Multiple failed login attempts',
            'user_agent': 'Mozilla/5.0',
            'location': 'Unknown',
            'blocked': False,
            'resolved': False
        }
    ]

def calculate_security_score():
    """Calculate security score"""
    return 8.7

def block_ip_address(ip):
    """Block IP address"""
    st.success(f"IP address {ip} blocked!")

def mark_security_event_safe(event_id):
    """Mark security event as safe"""
    st.success(f"Security event {event_id} marked as safe!")
