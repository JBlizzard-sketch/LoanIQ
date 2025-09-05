import streamlit as st
from auth import AuthManager

def show_login_page():
    """Display the login page"""
    st.title("🏦 LoanIQ Credit Scoring Platform")
    st.markdown("### Secure Login")
    
    auth_manager = AuthManager()
    
    # Login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("#### Enter your credentials")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_login, col_register = st.columns(2)
            
            with col_login:
                login_submitted = st.form_submit_button("🔐 Login", use_container_width=True)
            
            with col_register:
                show_register = st.form_submit_button("📝 Register", use_container_width=True)
        
        if login_submitted:
            if username and password:
                success, role = auth_manager.authenticate(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = role
                    st.session_state.page = 'homepage'
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.warning("Please enter both username and password")
        
        if show_register:
            st.session_state.show_register_form = True
            st.rerun()
    
    # Registration form
    if st.session_state.get('show_register_form', False):
        st.markdown("---")
        
        with col2:
            with st.form("register_form"):
                st.markdown("#### Create New Account")
                
                new_username = st.text_input("New Username", placeholder="Choose a username")
                new_password = st.text_input("New Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                col_submit, col_cancel = st.columns(2)
                
                with col_submit:
                    register_submitted = st.form_submit_button("✅ Create Account", use_container_width=True)
                
                with col_cancel:
                    cancel_register = st.form_submit_button("❌ Cancel", use_container_width=True)
            
            if register_submitted:
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            success, message = auth_manager.register_user(new_username, new_password)
                            if success:
                                st.success(message)
                                st.session_state.show_register_form = False
                                st.rerun()
                            else:
                                st.error(message)
                        else:
                            st.error("Password must be at least 6 characters long")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please fill in all fields")
            
            if cancel_register:
                st.session_state.show_register_form = False
                st.rerun()
    
    # Demo credentials info
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Demo Admin Account:**
        - Username: `admin`
        - Password: `Shady868`
        """)
    
    with col2:
        st.info("""
        **Demo Client Account:**
        - Username: `demo_client`
        - Password: `demo123`
        """)
