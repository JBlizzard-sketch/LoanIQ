import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pickle
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'loaniq_database.db'):
        """
        Initialize database manager for LoanIQ platform
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
        # Initialize database and create tables
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database and create necessary tables"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Create tables
            self._create_tables()
            
            logger.info(f"Database initialized successfully: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _create_tables(self):
        """Create all necessary database tables"""
        
        # Users table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'client',
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                settings TEXT  -- JSON string for user settings
            )
        ''')
        
        # Applications table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id TEXT UNIQUE NOT NULL,
                user_id INTEGER,
                applicant_data TEXT NOT NULL,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Predictions table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id TEXT NOT NULL,
                user_id INTEGER,
                model_name TEXT NOT NULL,
                model_version TEXT,
                risk_score REAL,
                default_probability REAL,
                risk_category TEXT,
                prediction_data TEXT,  -- JSON string with full prediction details
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (application_id) REFERENCES applications (application_id)
            )
        ''')
        
        # Model versions table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_data BLOB,  -- Pickled model
                metadata TEXT,    -- JSON string with model metadata
                performance_metrics TEXT,  -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                created_by INTEGER,
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        
        # Datasets table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                dataset_type TEXT,  -- 'uploaded', 'synthetic', 'api'
                data_path TEXT,     -- Path to stored data file
                schema_info TEXT,   -- JSON string with schema information
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                size_mb REAL,
                record_count INTEGER,
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        
        # System logs table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                log_level TEXT NOT NULL,
                component TEXT NOT NULL,
                user_id INTEGER,
                message TEXT NOT NULL,
                details TEXT,  -- JSON string with additional details
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Performance metrics table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,  -- 'response_time', 'throughput', 'error_rate', etc.
                metric_value REAL NOT NULL,
                component TEXT,
                additional_data TEXT  -- JSON string
            )
        ''')
        
        # API usage table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                response_time_ms REAL,
                status_code INTEGER,
                request_size_bytes INTEGER,
                response_size_bytes INTEGER,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Model training jobs table
        self._execute_query('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                dataset_id TEXT,
                config TEXT,  -- JSON string with training configuration
                status TEXT DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed'
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_by INTEGER,
                results TEXT,  -- JSON string with training results
                error_message TEXT,
                FOREIGN KEY (created_by) REFERENCES users (id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
            )
        ''')
        
        logger.info("Database tables created successfully")
    
    def _execute_query(self, query: str, params: tuple = None) -> sqlite3.Cursor:
        """
        Execute a SQL query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            SQLite cursor object
        """
        try:
            if params:
                cursor = self.connection.execute(query, params)
            else:
                cursor = self.connection.execute(query)
            
            self.connection.commit()
            return cursor
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            raise
    
    def _fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """
        Execute query and fetch all results as list of dictionaries
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of dictionaries representing rows
        """
        cursor = self._execute_query(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def _fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """
        Execute query and fetch one result as dictionary
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Dictionary representing row or None
        """
        cursor = self._execute_query(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # User management methods
    def create_user(self, username: str, password_hash: str, role: str = 'client', 
                   email: str = None, settings: Dict = None) -> int:
        """
        Create a new user
        
        Args:
            username: Unique username
            password_hash: Hashed password
            role: User role ('client' or 'admin')
            email: User email
            settings: User settings dictionary
            
        Returns:
            User ID
        """
        settings_json = json.dumps(settings) if settings else None
        
        cursor = self._execute_query('''
            INSERT INTO users (username, password_hash, role, email, settings)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, role, email, settings_json))
        
        user_id = cursor.lastrowid
        
        # Log user creation
        self.log_system_event('INFO', 'Authentication', 
                            f"User created: {username}", user_id=user_id)
        
        return user_id
    
    def get_user(self, username: str) -> Optional[Dict]:
        """
        Get user by username
        
        Args:
            username: Username to lookup
            
        Returns:
            User dictionary or None
        """
        user = self._fetch_one('SELECT * FROM users WHERE username = ?', (username,))
        
        if user and user['settings']:
            try:
                user['settings'] = json.loads(user['settings'])
            except json.JSONDecodeError:
                user['settings'] = {}
        
        return user
    
    def update_user_login(self, username: str):
        """Update user's last login timestamp"""
        self._execute_query('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP 
            WHERE username = ?
        ''', (username,))
        
        # Log login
        user = self.get_user(username)
        if user:
            self.log_system_event('INFO', 'Authentication', 
                                f"User logged in: {username}", user_id=user['id'])
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        users = self._fetch_all('SELECT * FROM users ORDER BY created_at DESC')
        
        for user in users:
            if user['settings']:
                try:
                    user['settings'] = json.loads(user['settings'])
                except json.JSONDecodeError:
                    user['settings'] = {}
        
        return users
    
    def update_user_status(self, username: str, is_active: bool):
        """Update user active status"""
        self._execute_query('''
            UPDATE users SET is_active = ? WHERE username = ?
        ''', (is_active, username))
        
        # Log status change
        user = self.get_user(username)
        if user:
            status = "activated" if is_active else "deactivated"
            self.log_system_event('INFO', 'User Management', 
                                f"User {status}: {username}", user_id=user['id'])
    
    # Application management methods
    def save_application(self, application_id: str, user_id: int, 
                        applicant_data: Dict, status: str = 'pending') -> int:
        """
        Save loan application
        
        Args:
            application_id: Unique application ID
            user_id: ID of user who created the application
            applicant_data: Application data dictionary
            status: Application status
            
        Returns:
            Application ID
        """
        applicant_data_json = json.dumps(applicant_data)
        
        cursor = self._execute_query('''
            INSERT INTO applications (application_id, user_id, applicant_data, status)
            VALUES (?, ?, ?, ?)
        ''', (application_id, user_id, applicant_data_json, status))
        
        # Log application creation
        self.log_system_event('INFO', 'Application', 
                            f"Application created: {application_id}", user_id=user_id)
        
        return cursor.lastrowid
    
    def get_application(self, application_id: str) -> Optional[Dict]:
        """Get application by ID"""
        app = self._fetch_one('''
            SELECT * FROM applications WHERE application_id = ?
        ''', (application_id,))
        
        if app and app['applicant_data']:
            try:
                app['applicant_data'] = json.loads(app['applicant_data'])
            except json.JSONDecodeError:
                app['applicant_data'] = {}
        
        return app
    
    def get_user_applications(self, user_id: int, limit: int = 100) -> List[Dict]:
        """Get applications for a specific user"""
        apps = self._fetch_all('''
            SELECT * FROM applications 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        for app in apps:
            if app['applicant_data']:
                try:
                    app['applicant_data'] = json.loads(app['applicant_data'])
                except json.JSONDecodeError:
                    app['applicant_data'] = {}
        
        return apps
    
    # Prediction management methods
    def save_prediction(self, application_id: str, user_id: int, model_name: str,
                       model_version: str, risk_score: float, default_probability: float,
                       risk_category: str, prediction_data: Dict) -> int:
        """
        Save prediction result
        
        Args:
            application_id: Application ID
            user_id: User ID
            model_name: Name of model used
            model_version: Version of model used
            risk_score: Calculated risk score
            default_probability: Default probability
            risk_category: Risk category
            prediction_data: Full prediction data dictionary
            
        Returns:
            Prediction ID
        """
        prediction_data_json = json.dumps(prediction_data)
        
        cursor = self._execute_query('''
            INSERT INTO predictions 
            (application_id, user_id, model_name, model_version, risk_score, 
             default_probability, risk_category, prediction_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (application_id, user_id, model_name, model_version, risk_score,
              default_probability, risk_category, prediction_data_json))
        
        # Log prediction
        self.log_system_event('INFO', 'ML Pipeline', 
                            f"Prediction created for {application_id}", user_id=user_id)
        
        return cursor.lastrowid
    
    def get_predictions(self, user_id: int = None, limit: int = 1000) -> List[Dict]:
        """Get predictions, optionally filtered by user"""
        if user_id:
            predictions = self._fetch_all('''
                SELECT * FROM predictions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
        else:
            predictions = self._fetch_all('''
                SELECT * FROM predictions 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
        
        for pred in predictions:
            if pred['prediction_data']:
                try:
                    pred['prediction_data'] = json.loads(pred['prediction_data'])
                except json.JSONDecodeError:
                    pred['prediction_data'] = {}
        
        return predictions
    
    def get_prediction_stats(self, user_id: int = None, days: int = 30) -> Dict:
        """Get prediction statistics"""
        date_filter = datetime.now() - timedelta(days=days)
        
        if user_id:
            base_query = '''
                SELECT COUNT(*) as count, AVG(risk_score) as avg_score,
                       AVG(default_probability) as avg_default_prob
                FROM predictions 
                WHERE user_id = ? AND created_at >= ?
            '''
            params = (user_id, date_filter)
        else:
            base_query = '''
                SELECT COUNT(*) as count, AVG(risk_score) as avg_score,
                       AVG(default_probability) as avg_default_prob
                FROM predictions 
                WHERE created_at >= ?
            '''
            params = (date_filter,)
        
        stats = self._fetch_one(base_query, params)
        
        # Get risk category distribution
        if user_id:
            risk_dist = self._fetch_all('''
                SELECT risk_category, COUNT(*) as count
                FROM predictions 
                WHERE user_id = ? AND created_at >= ?
                GROUP BY risk_category
            ''', (user_id, date_filter))
        else:
            risk_dist = self._fetch_all('''
                SELECT risk_category, COUNT(*) as count
                FROM predictions 
                WHERE created_at >= ?
                GROUP BY risk_category
            ''', (date_filter,))
        
        stats['risk_distribution'] = {item['risk_category']: item['count'] for item in risk_dist}
        
        return stats
    
    # Model version management methods
    def save_model_version(self, version_id: str, model_name: str, version: str,
                          model_data: Any, metadata: Dict, performance_metrics: Dict,
                          created_by: int, status: str = 'active') -> int:
        """
        Save model version
        
        Args:
            version_id: Unique version identifier
            model_name: Name of the model
            version: Version string
            model_data: Serialized model data
            metadata: Model metadata dictionary
            performance_metrics: Performance metrics dictionary
            created_by: User ID who created the version
            status: Version status
            
        Returns:
            Model version ID
        """
        model_blob = pickle.dumps(model_data)
        metadata_json = json.dumps(metadata)
        metrics_json = json.dumps(performance_metrics)
        
        cursor = self._execute_query('''
            INSERT INTO model_versions 
            (version_id, model_name, version, model_data, metadata, 
             performance_metrics, created_by, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (version_id, model_name, version, model_blob, metadata_json,
              metrics_json, created_by, status))
        
        # Log model version creation
        self.log_system_event('INFO', 'Model Management', 
                            f"Model version created: {version_id}", user_id=created_by)
        
        return cursor.lastrowid
    
    def get_model_version(self, version_id: str) -> Optional[Dict]:
        """Get model version by ID"""
        version = self._fetch_one('''
            SELECT * FROM model_versions WHERE version_id = ?
        ''', (version_id,))
        
        if version:
            # Deserialize model data
            if version['model_data']:
                try:
                    version['model_data'] = pickle.loads(version['model_data'])
                except Exception as e:
                    logger.error(f"Error deserializing model data: {e}")
                    version['model_data'] = None
            
            # Parse JSON fields
            for field in ['metadata', 'performance_metrics']:
                if version[field]:
                    try:
                        version[field] = json.loads(version[field])
                    except json.JSONDecodeError:
                        version[field] = {}
        
        return version
    
    def get_model_versions(self, model_name: str = None) -> List[Dict]:
        """Get model versions, optionally filtered by model name"""
        if model_name:
            versions = self._fetch_all('''
                SELECT * FROM model_versions 
                WHERE model_name = ? 
                ORDER BY created_at DESC
            ''', (model_name,))
        else:
            versions = self._fetch_all('''
                SELECT * FROM model_versions 
                ORDER BY created_at DESC
            ''')
        
        for version in versions:
            # Don't deserialize model_data in list view for performance
            version['model_data'] = None
            
            # Parse JSON fields
            for field in ['metadata', 'performance_metrics']:
                if version[field]:
                    try:
                        version[field] = json.loads(version[field])
                    except json.JSONDecodeError:
                        version[field] = {}
        
        return versions
    
    # Dataset management methods
    def save_dataset(self, dataset_id: str, name: str, description: str,
                    dataset_type: str, data_path: str, schema_info: Dict,
                    created_by: int, size_mb: float, record_count: int) -> int:
        """Save dataset information"""
        schema_json = json.dumps(schema_info)
        
        cursor = self._execute_query('''
            INSERT INTO datasets 
            (dataset_id, name, description, dataset_type, data_path, 
             schema_info, created_by, size_mb, record_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dataset_id, name, description, dataset_type, data_path,
              schema_json, created_by, size_mb, record_count))
        
        # Log dataset creation
        self.log_system_event('INFO', 'Data Management', 
                            f"Dataset created: {dataset_id}", user_id=created_by)
        
        return cursor.lastrowid
    
    def get_datasets(self, created_by: int = None) -> List[Dict]:
        """Get datasets, optionally filtered by creator"""
        if created_by:
            datasets = self._fetch_all('''
                SELECT * FROM datasets 
                WHERE created_by = ? 
                ORDER BY created_at DESC
            ''', (created_by,))
        else:
            datasets = self._fetch_all('''
                SELECT * FROM datasets 
                ORDER BY created_at DESC
            ''')
        
        for dataset in datasets:
            if dataset['schema_info']:
                try:
                    dataset['schema_info'] = json.loads(dataset['schema_info'])
                except json.JSONDecodeError:
                    dataset['schema_info'] = {}
        
        return datasets
    
    # System logging methods
    def log_system_event(self, log_level: str, component: str, message: str,
                        user_id: int = None, details: Dict = None,
                        ip_address: str = None, user_agent: str = None):
        """
        Log system event
        
        Args:
            log_level: Log level ('INFO', 'WARNING', 'ERROR', 'DEBUG')
            component: System component
            message: Log message
            user_id: Associated user ID
            details: Additional details dictionary
            ip_address: Client IP address
            user_agent: Client user agent
        """
        details_json = json.dumps(details) if details else None
        
        self._execute_query('''
            INSERT INTO system_logs 
            (log_level, component, user_id, message, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (log_level, component, user_id, message, details_json, ip_address, user_agent))
    
    def get_system_logs(self, log_level: str = None, component: str = None,
                       days: int = 7, limit: int = 1000) -> List[Dict]:
        """Get system logs with optional filtering"""
        date_filter = datetime.now() - timedelta(days=days)
        
        query = 'SELECT * FROM system_logs WHERE timestamp >= ?'
        params = [date_filter]
        
        if log_level:
            query += ' AND log_level = ?'
            params.append(log_level)
        
        if component:
            query += ' AND component = ?'
            params.append(component)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        logs = self._fetch_all(query, tuple(params))
        
        for log in logs:
            if log['details']:
                try:
                    log['details'] = json.loads(log['details'])
                except json.JSONDecodeError:
                    log['details'] = {}
        
        return logs
    
    # Performance monitoring methods
    def record_performance_metric(self, metric_type: str, metric_value: float,
                                 component: str = None, additional_data: Dict = None):
        """Record performance metric"""
        additional_json = json.dumps(additional_data) if additional_data else None
        
        self._execute_query('''
            INSERT INTO performance_metrics 
            (metric_type, metric_value, component, additional_data)
            VALUES (?, ?, ?, ?)
        ''', (metric_type, metric_value, component, additional_json))
    
    def get_performance_metrics(self, metric_type: str = None, component: str = None,
                               hours: int = 24) -> List[Dict]:
        """Get performance metrics"""
        date_filter = datetime.now() - timedelta(hours=hours)
        
        query = 'SELECT * FROM performance_metrics WHERE timestamp >= ?'
        params = [date_filter]
        
        if metric_type:
            query += ' AND metric_type = ?'
            params.append(metric_type)
        
        if component:
            query += ' AND component = ?'
            params.append(component)
        
        query += ' ORDER BY timestamp DESC'
        
        metrics = self._fetch_all(query, tuple(params))
        
        for metric in metrics:
            if metric['additional_data']:
                try:
                    metric['additional_data'] = json.loads(metric['additional_data'])
                except json.JSONDecodeError:
                    metric['additional_data'] = {}
        
        return metrics
    
    # API usage tracking methods
    def record_api_usage(self, user_id: int, endpoint: str, method: str,
                        response_time_ms: float, status_code: int,
                        request_size_bytes: int = None, response_size_bytes: int = None,
                        ip_address: str = None):
        """Record API usage"""
        self._execute_query('''
            INSERT INTO api_usage 
            (user_id, endpoint, method, response_time_ms, status_code,
             request_size_bytes, response_size_bytes, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, endpoint, method, response_time_ms, status_code,
              request_size_bytes, response_size_bytes, ip_address))
    
    def get_api_usage_stats(self, user_id: int = None, hours: int = 24) -> Dict:
        """Get API usage statistics"""
        date_filter = datetime.now() - timedelta(hours=hours)
        
        if user_id:
            base_query = '''
                SELECT COUNT(*) as total_requests,
                       AVG(response_time_ms) as avg_response_time,
                       SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
                FROM api_usage 
                WHERE user_id = ? AND timestamp >= ?
            '''
            params = (user_id, date_filter)
        else:
            base_query = '''
                SELECT COUNT(*) as total_requests,
                       AVG(response_time_ms) as avg_response_time,
                       SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
                FROM api_usage 
                WHERE timestamp >= ?
            '''
            params = (date_filter,)
        
        stats = self._fetch_one(base_query, params)
        
        # Calculate error rate
        if stats['total_requests'] > 0:
            stats['error_rate'] = stats['error_count'] / stats['total_requests']
        else:
            stats['error_rate'] = 0
        
        return stats
    
    # Training job management methods
    def create_training_job(self, job_id: str, model_name: str, dataset_id: str,
                           config: Dict, created_by: int) -> int:
        """Create training job"""
        config_json = json.dumps(config)
        
        cursor = self._execute_query('''
            INSERT INTO training_jobs 
            (job_id, model_name, dataset_id, config, created_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_id, model_name, dataset_id, config_json, created_by))
        
        # Log job creation
        self.log_system_event('INFO', 'Model Training', 
                            f"Training job created: {job_id}", user_id=created_by)
        
        return cursor.lastrowid
    
    def update_training_job_status(self, job_id: str, status: str,
                                  results: Dict = None, error_message: str = None):
        """Update training job status"""
        results_json = json.dumps(results) if results else None
        
        if status == 'running':
            self._execute_query('''
                UPDATE training_jobs 
                SET status = ?, started_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
            ''', (status, job_id))
        elif status in ['completed', 'failed']:
            self._execute_query('''
                UPDATE training_jobs 
                SET status = ?, completed_at = CURRENT_TIMESTAMP, 
                    results = ?, error_message = ?
                WHERE job_id = ?
            ''', (status, results_json, error_message, job_id))
        else:
            self._execute_query('''
                UPDATE training_jobs SET status = ? WHERE job_id = ?
            ''', (status, job_id))
    
    def get_training_jobs(self, created_by: int = None, status: str = None) -> List[Dict]:
        """Get training jobs"""
        query = 'SELECT * FROM training_jobs WHERE 1=1'
        params = []
        
        if created_by:
            query += ' AND created_by = ?'
            params.append(created_by)
        
        if status:
            query += ' AND status = ?'
            params.append(status)
        
        query += ' ORDER BY created_at DESC'
        
        jobs = self._fetch_all(query, tuple(params))
        
        for job in jobs:
            for field in ['config', 'results']:
                if job[field]:
                    try:
                        job[field] = json.loads(job[field])
                    except json.JSONDecodeError:
                        job[field] = {}
        
        return jobs
    
    # Utility methods
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        tables = ['users', 'applications', 'predictions', 'model_versions', 
                 'datasets', 'system_logs', 'performance_metrics', 'api_usage', 'training_jobs']
        
        for table in tables:
            count = self._fetch_one(f'SELECT COUNT(*) as count FROM {table}')
            stats[f'{table}_count'] = count['count']
        
        # Database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        stats['database_size_mb'] = db_size / (1024 * 1024)
        
        return stats
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean old logs
        self._execute_query('''
            DELETE FROM system_logs WHERE timestamp < ?
        ''', (cutoff_date,))
        
        # Clean old performance metrics
        self._execute_query('''
            DELETE FROM performance_metrics WHERE timestamp < ?
        ''', (cutoff_date,))
        
        # Clean old API usage records
        self._execute_query('''
            DELETE FROM api_usage WHERE timestamp < ?
        ''', (cutoff_date,))
        
        logger.info(f"Cleaned up data older than {days} days")
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        import shutil
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            
            # Log backup
            self.log_system_event('INFO', 'Database', 
                                f"Database backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close_connection()
