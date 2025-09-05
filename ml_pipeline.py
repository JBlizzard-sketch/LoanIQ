import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.target_name = 'default'
        self.model_directory = 'saved_models'
        self.performance_history = []
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Load existing models if available
        self._load_existing_models()
    
    def _initialize_models(self):
        """Initialize all ML models"""
        self.model_configs = {
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                ),
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ),
                'hyperparameters': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'hyperparameters': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 6, 9]
                }
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    random_state=42,
                    probability=True
                ),
                'hyperparameters': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
    
    def _load_existing_models(self):
        """Load existing trained models"""
        for model_name in self.model_configs.keys():
            model_path = os.path.join(self.model_directory, f'{model_name.lower()}_model.pkl')
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        self.models[model_name] = model_data['model']
                        if 'scaler' in model_data:
                            self.scalers[model_name] = model_data['scaler']
                        if 'encoders' in model_data:
                            self.encoders[model_name] = model_data['encoders']
                        if 'feature_names' in model_data:
                            self.feature_names = model_data['feature_names']
                except Exception as e:
                    print(f"Error loading {model_name}: {str(e)}")
    
    def preprocess_data(self, data):
        """Preprocess data for training or prediction"""
        df = data.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill missing numeric values with median
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Encode categorical variables
        encoded_columns = []
        for col in categorical_columns:
            if col != self.target_name:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoded_columns.append(f'{col}_encoded')
                
                # Store encoder for later use
                if 'encoders' not in self.__dict__:
                    self.encoders = {}
                self.encoders[col] = le
        
        # Drop original categorical columns (except target)
        columns_to_drop = [col for col in categorical_columns if col != self.target_name]
        df = df.drop(columns=columns_to_drop)
        
        return df
    
    def train_model(self, data, model_name='XGBoost', hyperparameter_tuning=False, cross_validation=True):
        """Train a specific model"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Separate features and target
        if self.target_name in processed_data.columns:
            X = processed_data.drop(columns=[self.target_name])
            y = processed_data[self.target_name]
        else:
            raise ValueError(f"Target column '{self.target_name}' not found in data")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for certain models
        if model_name in ['LogisticRegression', 'SVM']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Get model
        model = self.model_configs[model_name]['model']
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grid = self.model_configs[model_name]['hyperparameters']
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        if cross_validation:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
        
        # Store model
        self.models[model_name] = model
        
        # Save model to disk
        self._save_model(model_name, model, metrics)
        
        # Update performance history
        performance_record = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        self.performance_history.append(performance_record)
        
        return metrics
    
    def train_all_models(self, data, hyperparameter_tuning=False):
        """Train all available models"""
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                print(f"Training {model_name}...")
                metrics = self.train_model(data, model_name, hyperparameter_tuning)
                results[model_name] = metrics
                print(f"{model_name} - AUC: {metrics['auc_score']:.3f}")
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_single(self, input_data):
        """Make prediction for a single instance"""
        # Get the best performing model
        best_model_name = self.get_best_model()
        
        if best_model_name not in self.models:
            # Use a simple rule-based prediction if no trained model
            return self._rule_based_prediction(input_data)
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocess
        df_processed = self._preprocess_single(df, best_model_name)
        
        # Get model and make prediction
        model = self.models[best_model_name]
        
        # Scale if necessary
        if best_model_name in self.scalers:
            df_processed = self.scalers[best_model_name].transform(df_processed)
        
        # Predict
        default_prob = model.predict_proba(df_processed)[0, 1]
        
        # Calculate risk score (inverse of default probability, scaled to credit score range)
        risk_score = int((1 - default_prob) * 550 + 300)  # Scale to 300-850 range
        
        # Determine risk category
        if risk_score >= 750:
            risk_category = "Low Risk"
        elif risk_score >= 650:
            risk_category = "Medium Risk"
        elif risk_score >= 550:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        return {
            'risk_score': risk_score,
            'default_probability': default_prob,
            'risk_category': risk_category,
            'model_used': best_model_name
        }
    
    def predict_batch(self, data):
        """Make predictions for batch data"""
        results = []
        
        for idx, row in data.iterrows():
            try:
                prediction = self.predict_single(row.to_dict())
                prediction['row_id'] = idx
                results.append(prediction)
            except Exception as e:
                results.append({
                    'row_id': idx,
                    'error': str(e),
                    'risk_score': 500,
                    'default_probability': 0.5,
                    'risk_category': 'Unknown'
                })
        
        return pd.DataFrame(results)
    
    def _preprocess_single(self, df, model_name):
        """Preprocess single instance for prediction"""
        df_processed = df.copy()
        
        # Handle categorical encoding
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col in self.encoders:
                # Handle unseen categories
                try:
                    df_processed[f'{col}_encoded'] = self.encoders[col].transform(df_processed[col].astype(str))
                except ValueError:
                    # If category not seen during training, use most frequent class
                    df_processed[f'{col}_encoded'] = 0
        
        # Drop original categorical columns
        df_processed = df_processed.drop(columns=categorical_columns.tolist())
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df_processed.columns)
        for feature in missing_features:
            df_processed[feature] = 0  # Fill missing features with 0
        
        # Reorder columns to match training
        df_processed = df_processed[self.feature_names]
        
        return df_processed
    
    def _rule_based_prediction(self, input_data):
        """Simple rule-based prediction when no trained model is available"""
        # Simple credit scoring rules
        score = 650  # Base score
        
        # Age factor
        age = input_data.get('age', 35)
        if age < 25:
            score -= 20
        elif age > 55:
            score += 20
        
        # Income factor
        income = input_data.get('annual_income', 50000)
        if income > 100000:
            score += 50
        elif income < 30000:
            score -= 50
        
        # Credit score factor
        credit_score = input_data.get('credit_score', 650)
        score = score + (credit_score - 650) * 0.5
        
        # Debt-to-income factor
        dti = input_data.get('debt_to_income_ratio', 0.3)
        if dti > 0.4:
            score -= 40
        elif dti < 0.2:
            score += 20
        
        # Ensure score is within range
        score = max(300, min(850, int(score)))
        
        # Calculate default probability
        default_prob = max(0, min(1, (850 - score) / 550))
        
        # Risk category
        if score >= 750:
            risk_category = "Low Risk"
        elif score >= 650:
            risk_category = "Medium Risk"
        elif score >= 550:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        return {
            'risk_score': score,
            'default_probability': default_prob,
            'risk_category': risk_category,
            'model_used': 'Rule-based'
        }
    
    def get_best_model(self):
        """Get the name of the best performing model"""
        if not self.performance_history:
            return 'XGBoost'  # Default
        
        best_model = max(
            self.performance_history,
            key=lambda x: x['metrics'].get('auc_score', 0)
        )
        
        return best_model['model_name']
    
    def get_active_model(self):
        """Get the currently active model"""
        best_model_name = self.get_best_model()
        return self.models.get(best_model_name)
    
    def get_models_performance(self):
        """Get performance metrics for all models"""
        performance_data = []
        
        for model_name in self.model_configs.keys():
            # Find latest performance for this model
            model_performances = [
                p for p in self.performance_history 
                if p['model_name'] == model_name
            ]
            
            if model_performances:
                latest_perf = max(model_performances, key=lambda x: x['timestamp'])
                metrics = latest_perf['metrics']
                
                performance_data.append({
                    'model_name': model_name,
                    'auc_score': metrics.get('auc_score', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0),
                    'last_trained': latest_perf['timestamp']
                })
            else:
                # Default values for untrained models
                performance_data.append({
                    'model_name': model_name,
                    'auc_score': 0.5,
                    'accuracy': 0.5,
                    'precision': 0.5,
                    'recall': 0.5,
                    'f1_score': 0.5,
                    'last_trained': 'Not trained'
                })
        
        return performance_data
    
    def get_detailed_models_info(self):
        """Get detailed information about all models"""
        detailed_info = []
        
        for model_name in self.model_configs.keys():
            # Find latest performance for this model
            model_performances = [
                p for p in self.performance_history 
                if p['model_name'] == model_name
            ]
            
            if model_performances:
                latest_perf = max(model_performances, key=lambda x: x['timestamp'])
                metrics = latest_perf['metrics']
                
                info = {
                    'name': model_name,
                    'version': '1.0',
                    'auc_score': metrics.get('auc_score', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'trained_date': latest_perf['timestamp'][:10],
                    'training_samples': latest_perf.get('training_samples', 0),
                    'n_features': len(self.feature_names)
                }
                
                # Add feature importance if available
                if model_name in self.models:
                    model = self.models[model_name]
                    if hasattr(model, 'feature_importances_'):
                        importance_dict = dict(zip(
                            self.feature_names, 
                            model.feature_importances_
                        ))
                        # Sort by importance and take top 10
                        top_features = sorted(
                            importance_dict.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:10]
                        info['feature_importance'] = dict(top_features)
                
                detailed_info.append(info)
        
        return detailed_info
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        fig = go.Figure()
        
        # Add diagonal line for random classifier
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random Classifier'
        ))
        
        # Add ROC curves for each model (simulated data for now)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, model_name in enumerate(self.model_configs.keys()):
            # Generate sample ROC curve data
            fpr = np.linspace(0, 1, 100)
            
            # Get AUC score for model
            performance = self.get_models_performance()
            model_perf = next((p for p in performance if p['model_name'] == model_name), None)
            auc = model_perf['auc_score'] if model_perf else 0.5
            
            # Generate TPR based on AUC
            tpr = np.minimum(fpr * 2, 1) * auc + fpr * (1 - auc)
            tpr = np.maximum(tpr, fpr)  # Ensure TPR >= FPR
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc:.3f})',
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700, height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for the best model"""
        # Generate sample confusion matrix data
        best_model = self.get_best_model()
        
        # Sample confusion matrix (would be calculated from actual predictions)
        cm = np.array([[850, 120], [90, 140]])
        
        fig = px.imshow(
            cm, 
            text_auto=True, 
            aspect="auto",
            title=f'Confusion Matrix - {best_model}',
            labels=dict(x="Predicted", y="Actual"),
            x=['No Default', 'Default'],
            y=['No Default', 'Default']
        )
        
        return fig
    
    def plot_feature_importance_comparison(self):
        """Plot feature importance comparison across models"""
        fig = go.Figure()
        
        # Sample feature importance data
        features = ['credit_score', 'annual_income', 'debt_to_income_ratio', 'age', 'employment_length']
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, model_name in enumerate(['XGBoost', 'LightGBM', 'RandomForest']):
            # Generate sample importance values
            np.random.seed(i)
            importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
            
            fig.add_trace(go.Bar(
                x=features,
                y=importance,
                name=model_name,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Feature Importance Comparison',
            xaxis_title='Features',
            yaxis_title='Importance',
            barmode='group',
            height=500
        )
        
        return fig
    
    def _save_model(self, model_name, model, metrics):
        """Save trained model to disk"""
        model_data = {
            'model': model,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'feature_names': self.feature_names
        }
        
        # Add scaler if exists
        if model_name in self.scalers:
            model_data['scaler'] = self.scalers[model_name]
        
        # Add encoders if exist
        if hasattr(self, 'encoders') and self.encoders:
            model_data['encoders'] = self.encoders
        
        # Save to pickle file
        model_path = os.path.join(self.model_directory, f'{model_name.lower()}_model.pkl')
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model {model_name} saved successfully")
        except Exception as e:
            print(f"Error saving model {model_name}: {str(e)}")
    
    def retrain_model(self, model_name, data):
        """Retrain a specific model with new data"""
        return self.train_model(data, model_name, hyperparameter_tuning=True)
    
    def get_model_version_info(self):
        """Get version information for all models"""
        version_info = []
        
        for model_name in self.model_configs.keys():
            model_path = os.path.join(self.model_directory, f'{model_name.lower()}_model.pkl')
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    version_info.append({
                        'model_name': model_name,
                        'version': '1.0',
                        'created_date': model_data.get('timestamp', 'Unknown'),
                        'metrics': model_data.get('metrics', {}),
                        'file_path': model_path
                    })
                except Exception as e:
                    print(f"Error reading model {model_name}: {str(e)}")
        
        return version_info
