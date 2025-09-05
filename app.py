# File: app.py

import streamlit as st
from auth import AuthenticationManager
from database.db_manager import DatabaseManager

# PATCH START: Controlled routing by role

def route_to_page():
    """Route users to the right page based on session state & role."""

    # If not logged in
    if "user" not in st.session_state or not st.session_state["user"]:
        st.switch_page("pages/login.py")
        return

    user = st.session_state["user"]
    role = st.session_state.get("role", "client")

    # Role-based routing
    if role == "admin":
        # Admin sees everything
        page_options = {
            "Dashboard": "pages/client_dashboard.py",
            "Admin Sandbox": "pages/admin_sandbox.py",
        }
    else:
        # Clients only see their dashboard
        page_options = {
            "Dashboard": "pages/client_dashboard.py",
        }

    # Sidebar navigation
    st.sidebar.title("📂 Navigation")
    choice = st.sidebar.radio("Go to:", list(page_options.keys()))

    # Switch to selected page
    st.switch_page(page_options[choice])

# PATCH END
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Import our custom modules
from auth import AuthManager
from pages.login import show_login_page
from pages.client_dashboard import show_client_dashboard
from pages.admin_sandbox import show_admin_sandbox
from models.ml_pipeline import MLPipeline
from database.db_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="LoanIQ Credit Scoring Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'ml_pipeline' not in st.session_state:
        st.session_state.ml_pipeline = MLPipeline()
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

def show_homepage():
    """Show the main homepage with navigation"""
    st.title("🏦 LoanIQ Credit Scoring Platform")
    st.markdown("### Welcome to the comprehensive credit risk analysis solution")
    
    # User info in sidebar
    with st.sidebar:
        st.success(f"Logged in as: **{st.session_state.username}**")
        st.info(f"Role: **{st.session_state.user_role.title()}**")
        
        st.markdown("---")
        
        # Navigation based on role
        if st.session_state.user_role == 'admin':
            st.markdown("### 🔧 Admin Options")
            if st.button("🎛️ Admin Sandbox", use_container_width=True):
                st.session_state.page = 'admin_sandbox'
                st.rerun()
        
        st.markdown("### 📊 Client Options")
        if st.button("📈 Dashboard", use_container_width=True):
            st.session_state.page = 'client_dashboard'
            st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_role = None
            st.session_state.page = 'login'
            st.rerun()
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎯 Credit Scoring
        - Advanced ML models (XGBoost, LightGBM, etc.)
        - Real-time risk assessment
        - SHAP explainability
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Analytics Dashboard
        - Interactive visualizations
        - Performance metrics
        - What-if simulations
        """)
    
    with col3:
        st.markdown("""
        #### 🔄 Model Management
        - Version control
        - Auto-retraining
        - Stress testing
        """)
    
    # Quick stats
    st.markdown("---")
    st.markdown("### 📈 Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Models", "6", "↗ 1")
    
    with col2:
        st.metric("Predictions Today", "1,247", "↗ 23%")
    
    with col3:
        st.metric("Model Accuracy", "84.2%", "↗ 2.1%")
    
    with col4:
        st.metric("Active Users", "89", "↗ 12")
    
    # Recent activity
    st.markdown("### 🕐 Recent Activity")
    
    activity_data = {
        'Time': ['2 minutes ago', '15 minutes ago', '1 hour ago', '3 hours ago'],
        'Activity': [
            'New prediction request processed',
            'Model XGBoost v2.1 deployed',
            'Synthetic data generated (1000 records)',
            'Weekly model performance report'
        ],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '📊 Info']
    }
    
    df_activity = pd.DataFrame(activity_data)
    st.dataframe(df_activity, use_container_width=True, hide_index=True)

def main():
    """Main application logic"""
    initialize_session_state()
    
    # Initialize page state
    if 'page' not in st.session_state:
        if st.session_state.authenticated:
            st.session_state.page = 'homepage'
        else:
            st.session_state.page = 'login'
    
    # Route to appropriate page
    if not st.session_state.authenticated:
        show_login_page()
    else:
        if st.session_state.page == 'homepage':
            show_homepage()
        elif st.session_state.page == 'client_dashboard':
            show_client_dashboard()
        elif st.session_state.page == 'admin_sandbox':
            if st.session_state.user_role == 'admin':
                show_admin_sandbox()
            else:
                st.error("Access denied. Admin privileges required.")
                if st.button("← Back to Homepage"):
                    st.session_state.page = 'homepage'
                    st.rerun()

if __name__ == "__main__":
    main()
