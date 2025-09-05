# Overview

LoanIQ is a comprehensive credit scoring platform built with Streamlit that provides machine learning-powered loan risk assessment capabilities. The system features role-based access control with separate interfaces for administrators and clients, supporting the complete ML lifecycle from data processing and model training to predictions and explainability. The platform includes synthetic data generation, model versioning, and advanced analytics capabilities for credit risk evaluation.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with multi-page architecture
- **Navigation**: Role-based page routing with session state management
- **UI Components**: Plotly for interactive visualizations and charts
- **Layout**: Wide layout with expandable sidebar for navigation

## Backend Architecture
- **Application Structure**: Modular design with separate modules for authentication, database, ML pipeline, and utilities
- **Session Management**: Streamlit session state for maintaining user authentication and application state
- **Data Processing**: Dedicated data processing utilities with validation and preprocessing capabilities
- **Model Pipeline**: Comprehensive ML pipeline supporting multiple algorithms (XGBoost, LightGBM, Random Forest, Logistic Regression)

## Authentication & Authorization
- **Authentication**: Custom AuthManager class with SHA-256 password hashing
- **User Roles**: Two-tier role system (admin/client) with different access levels
- **Session Persistence**: File-based user storage with JSON format
- **Default Credentials**: Pre-configured admin and demo client accounts

## Data Storage Solutions
- **Primary Database**: SQLite database managed through custom DatabaseManager class
- **User Data**: JSON file storage for authentication credentials
- **Model Persistence**: Pickle serialization for trained ML models
- **Version Control**: Custom model versioning system with metadata tracking

## Machine Learning Architecture
- **Multi-Model Support**: Integration of XGBoost, LightGBM, Random Forest, and traditional algorithms
- **Feature Engineering**: Automated preprocessing pipeline with scaling and encoding
- **Model Explainability**: SHAP integration for model interpretability and feature importance
- **Synthetic Data**: Built-in synthetic data generation for testing and demos
- **Performance Tracking**: Model performance monitoring and comparison capabilities

## Key Design Patterns
- **Singleton Pattern**: Database and ML pipeline instances maintained in session state
- **Factory Pattern**: Dynamic model initialization and management
- **Strategy Pattern**: Multiple model algorithms with unified interface
- **Observer Pattern**: Performance tracking and logging system

# External Dependencies

## Core Framework Dependencies
- **streamlit**: Primary web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing operations
- **plotly**: Interactive data visualization (express and graph_objects)

## Machine Learning Libraries
- **scikit-learn**: Traditional ML algorithms and preprocessing utilities
- **xgboost**: Gradient boosting framework for high-performance models
- **lightgbm**: Microsoft's gradient boosting framework
- **shap**: Model explainability and feature importance analysis

## Data Storage & Processing
- **sqlite3**: Built-in SQLite database connectivity
- **pickle**: Python object serialization for model persistence
- **json**: Configuration and user data storage
- **hashlib**: Cryptographic hashing for password security

## Utility Libraries
- **datetime**: Date and time handling for timestamps and scheduling
- **os**: Operating system interface for file management
- **logging**: Application logging and error tracking
- **scipy**: Scientific computing utilities for statistical operations

## Development & Testing
- **warnings**: Warning management during development
- **typing**: Type hints for better code documentation
- **re**: Regular expressions for data validation

The system is designed as a self-contained platform with minimal external service dependencies, focusing on local processing and storage capabilities suitable for both development and production deployment scenarios.