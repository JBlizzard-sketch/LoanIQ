# File: pages/login.py

import streamlit as st
from auth import AuthenticationManager
from database.db_manager import DatabaseManager

# PATCH START: Login & Registration page

def show_login_page():
    """Render the login / registration UI and handle authentication."""

    st.title("🔐 LoanIQ Login")

    # Setup managers
    db_manager = DatabaseManager()
    auth_manager = AuthenticationManager(db_manager)

    # Tabs: Login | Register
    tabs = st.tabs(["Login", "Register"])

    with tabs[0]:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            user = auth_manager.login_user(email, password)
            if user:
                st.session_state["user"] = user
                st.session_state["role"] = user["role"]
                st.success("Login successful! Redirecting...")

                # Redirect based on role
                if user["role"] == "admin":
                    st.switch_page("pages/admin_sandbox.py")
                else:
                    st.switch_page("pages/client_dashboard.py")
            else:
                st.error("Invalid credentials.")

    with tabs[1]:
        st.subheader("Register")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register", key="register_button"):
            success = auth_manager.register_user(email, password)
            if success:
                st.success("Registration successful! Logging you in...")
                # Auto-login and redirect
                user = auth_manager.login_user(email, password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["role"] = user["role"]
                    if user["role"] == "admin":
                        st.switch_page("pages/admin_sandbox.py")
                    else:
                        st.switch_page("pages/client_dashboard.py")
            else:
                st.error("Registration failed. Try a different email.")

    st.divider()

    # If already logged in
    if "user" in st.session_state and st.session_state["user"]:
        user = st.session_state["user"]
        st.info(f"Logged in as {user['email']} ({user['role']})")
        if st.button("Logout", key="logout_button"):
            auth_manager.logout_user()
            st.session_state.clear()
            st.success("Logged out.")
            st.experimental_rerun()

# PATCH END
import streamlit as st
import hashlib
import os
from datetime import datetime, timedelta
import json

class AuthManager:
    def __init__(self):
        self.users_file = "users.json"
        self.initialize_default_users()
    
    def initialize_default_users(self):
        """Initialize default users if users file doesn't exist"""
        default_users = {
            "admin": {
                "password_hash": self.hash_password("Shady868"),
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            },
            "demo_client": {
                "password_hash": self.hash_password("demo123"),
                "role": "client",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
        }
        
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump(default_users, f, indent=2)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_users(self, users):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def authenticate(self, username, password):
        """Authenticate user credentials"""
        users = self.load_users()
        
        if username not in users:
            return False, None
        
        password_hash = self.hash_password(password)
        if users[username]["password_hash"] == password_hash:
            # Update last login
            users[username]["last_login"] = datetime.now().isoformat()
            self.save_users(users)
            return True, users[username]["role"]
        
        return False, None
    
    def register_user(self, username, password, role="client"):
        """Register a new user"""
        users = self.load_users()
        
        if username in users:
            return False, "Username already exists"
        
        users[username] = {
            "password_hash": self.hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        self.save_users(users)
        return True, "User registered successfully"
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        old_password_hash = self.hash_password(old_password)
        if users[username]["password_hash"] != old_password_hash:
            return False, "Current password is incorrect"
        
        users[username]["password_hash"] = self.hash_password(new_password)
        self.save_users(users)
        return True, "Password changed successfully"
    
    def get_user_role(self, username):
        """Get user role"""
        users = self.load_users()
        return users.get(username, {}).get("role", None)
