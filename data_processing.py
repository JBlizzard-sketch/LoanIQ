import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.data_schema = self._get_default_schema()
        self.preprocessing_history = []
        
    def _get_default_schema(self):
        """Get default data schema for loan applications"""
        return {
            'required_fields': [
                'age', 'annual_income', 'credit_score', 'debt_to_income_ratio',
                'employment_length', 'home_ownership', 'loan_purpose', 'loan_amount'
            ],
            'optional_fields': [
                'monthly_income', 'monthly_debt_payment', 'credit_history_length',
                'number_of_credit_lines', 'revolving_credit_balance', 'total_credit_limit'
            ],
            'categorical_fields': [
                'employment_length', 'home_ownership', 'loan_purpose', 'loan_grade',
                'verification_status', 'loan_status'
            ],
            'numeric_fields': [
                'age', 'annual_income', 'monthly_income', 'credit_score', 
                'debt_to_income_ratio', 'loan_amount', 'interest_rate',
                'installment', 'credit_history_length', 'delinq_2yrs',
                'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
                'revol_util', 'total_acc'
            ],
            'validation_rules': {
                'age': {'min': 18, 'max': 100},
                'credit_score': {'min': 300, 'max': 850},
                'annual_income': {'min': 0, 'max': 10000000},
                'debt_to_income_ratio': {'min': 0, 'max': 1},
                'loan_amount': {'min': 500, 'max': 100000},
                'employment_length': {
                    'allowed_values': ['< 1 year', '1-2 years', '3-5 years', 
                                     '5-10 years', '10+ years', 'Unknown']
                },
                'home_ownership': {
                    'allowed_values': ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data against schema"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'data_quality_score': 100,
            'missing_data_summary': {},
            'outlier_summary': {}
        }
        
        # Check required fields
        missing_required = set(self.data_schema['required_fields']) - set(data.columns)
        if missing_required:
            validation_results['errors'].append(
                f"Missing required fields: {', '.join(missing_required)}"
            )
            validation_results['valid'] = False
        
        # Check data types and ranges
        for column in data.columns:
            if column in self.data_schema['validation_rules']:
                rules = self.data_schema['validation_rules'][column]
                
                # Check numeric ranges
                if 'min' in rules and 'max' in rules:
                    numeric_data = pd.to_numeric(data[column], errors='coerce')
                    out_of_range = (
                        (numeric_data < rules['min']) | 
                        (numeric_data > rules['max'])
                    ).sum()
                    
                    if out_of_range > 0:
                        validation_results['warnings'].append(
                            f"{column}: {out_of_range} values outside valid range "
                            f"[{rules['min']}, {rules['max']}]"
                        )
                        validation_results['data_quality_score'] -= 5
                
                # Check categorical values
                if 'allowed_values' in rules:
                    invalid_values = set(data[column].dropna().unique()) - set(rules['allowed_values'])
                    if invalid_values:
                        validation_results['warnings'].append(
                            f"{column}: Invalid values found: {', '.join(map(str, invalid_values))}"
                        )
                        validation_results['data_quality_score'] -= 3
        
        # Check for missing data
        missing_summary = data.isnull().sum()
        if missing_summary.sum() > 0:
            validation_results['missing_data_summary'] = missing_summary[missing_summary > 0].to_dict()
            missing_percentage = (missing_summary.sum() / (len(data) * len(data.columns))) * 100
            
            if missing_percentage > 20:
                validation_results['errors'].append(
                    f"High percentage of missing data: {missing_percentage:.1f}%"
                )
                validation_results['valid'] = False
            elif missing_percentage > 5:
                validation_results['warnings'].append(
                    f"Moderate missing data: {missing_percentage:.1f}%"
                )
                validation_results['data_quality_score'] -= 10
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(
                f"Found {duplicate_count} duplicate records"
            )
            validation_results['data_quality_score'] -= 5
        
        # Detect outliers
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for column in numeric_columns:
            if column in data.columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
                if outliers > 0:
                    outlier_summary[column] = outliers
        
        if outlier_summary:
            validation_results['outlier_summary'] = outlier_summary
            total_outliers = sum(outlier_summary.values())
            outlier_percentage = (total_outliers / len(data)) * 100
            
            if outlier_percentage > 10:
                validation_results['warnings'].append(
                    f"High number of outliers detected: {outlier_percentage:.1f}%"
                )
                validation_results['data_quality_score'] -= 8
        
        # Generate suggestions
        if validation_results['missing_data_summary']:
            validation_results['suggestions'].append(
                "Consider using imputation techniques for missing values"
            )
        
        if validation_results['outlier_summary']:
            validation_results['suggestions'].append(
                "Review outliers - they might be data entry errors or legitimate edge cases"
            )
        
        if duplicate_count > 0:
            validation_results['suggestions'].append(
                "Remove duplicate records before training"
            )
        
        # Overall data quality assessment
        if validation_results['data_quality_score'] < 70:
            validation_results['suggestions'].append(
                "Consider comprehensive data cleaning before using for model training"
            )
        
        return validation_results
    
    def preprocess_data(self, data: pd.DataFrame, target_column: str = None,
                       imputation_strategy: str = 'median', 
                       scaling_method: str = 'standard',
                       handle_outliers: bool = True,
                       feature_selection: bool = False) -> pd.DataFrame:
        """Comprehensive data preprocessing pipeline"""
        
        processed_data = data.copy()
        preprocessing_log = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': data.shape,
            'steps_performed': []
        }
        
        # Step 1: Handle missing values
        processed_data, missing_log = self._handle_missing_values(
            processed_data, strategy=imputation_strategy
        )
        preprocessing_log['steps_performed'].append(missing_log)
        
        # Step 2: Handle outliers
        if handle_outliers:
            processed_data, outlier_log = self._handle_outliers(processed_data)
            preprocessing_log['steps_performed'].append(outlier_log)
        
        # Step 3: Encode categorical variables
        processed_data, encoding_log = self._encode_categorical_variables(processed_data)
        preprocessing_log['steps_performed'].append(encoding_log)
        
        # Step 4: Feature engineering
        processed_data, feature_log = self._engineer_features(processed_data)
        preprocessing_log['steps_performed'].append(feature_log)
        
        # Step 5: Scale numerical features
        if scaling_method:
            processed_data, scaling_log = self._scale_features(
                processed_data, method=scaling_method, target_column=target_column
            )
            preprocessing_log['steps_performed'].append(scaling_log)
        
        # Step 6: Feature selection
        if feature_selection and target_column and target_column in processed_data.columns:
            processed_data, selection_log = self._select_features(
                processed_data, target_column=target_column
            )
            preprocessing_log['steps_performed'].append(selection_log)
        
        preprocessing_log['output_shape'] = processed_data.shape
        self.preprocessing_history.append(preprocessing_log)
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str = 'median') -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values in the dataset"""
        processed_data = data.copy()
        log = {'step': 'missing_value_imputation', 'strategy': strategy, 'columns_processed': []}
        
        # Separate numeric and categorical columns
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        if len(numeric_columns) > 0:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                self.imputers['numeric_knn'] = imputer
            else:
                imputer = SimpleImputer(strategy=strategy)
                self.imputers['numeric_simple'] = imputer
            
            # Store columns with missing values
            numeric_missing = numeric_columns[processed_data[numeric_columns].isnull().any()]
            if len(numeric_missing) > 0:
                processed_data[numeric_columns] = imputer.fit_transform(processed_data[numeric_columns])
                log['columns_processed'].extend(numeric_missing.tolist())
        
        # Handle categorical missing values
        if len(categorical_columns) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            self.imputers['categorical'] = cat_imputer
            
            categorical_missing = categorical_columns[processed_data[categorical_columns].isnull().any()]
            if len(categorical_missing) > 0:
                processed_data[categorical_columns] = cat_imputer.fit_transform(processed_data[categorical_columns])
                log['columns_processed'].extend(categorical_missing.tolist())
        
        log['missing_values_filled'] = len(log['columns_processed'])
        
        return processed_data, log
    
    def _handle_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> Tuple[pd.DataFrame, Dict]:
        """Handle outliers in numerical columns"""
        processed_data = data.copy()
        log = {'step': 'outlier_handling', 'method': method, 'columns_processed': [], 'outliers_capped': 0}
        
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = processed_data[column].quantile(0.25)
                Q3 = processed_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (processed_data[column] < lower_bound) | (processed_data[column] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Cap outliers instead of removing them
                    processed_data.loc[processed_data[column] < lower_bound, column] = lower_bound
                    processed_data.loc[processed_data[column] > upper_bound, column] = upper_bound
                    
                    log['columns_processed'].append(column)
                    log['outliers_capped'] += outliers_count
            
            elif method == 'zscore':
                z_scores = np.abs((processed_data[column] - processed_data[column].mean()) / processed_data[column].std())
                outliers_mask = z_scores > 3
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Cap at 3 standard deviations
                    mean_val = processed_data[column].mean()
                    std_val = processed_data[column].std()
                    upper_bound = mean_val + 3 * std_val
                    lower_bound = mean_val - 3 * std_val
                    
                    processed_data.loc[processed_data[column] > upper_bound, column] = upper_bound
                    processed_data.loc[processed_data[column] < lower_bound, column] = lower_bound
                    
                    log['columns_processed'].append(column)
                    log['outliers_capped'] += outliers_count
        
        return processed_data, log
    
    def _encode_categorical_variables(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables"""
        processed_data = data.copy()
        log = {'step': 'categorical_encoding', 'columns_processed': [], 'encoding_mappings': {}}
        
        categorical_columns = processed_data.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column in self.data_schema['categorical_fields']:
                # Use label encoding for ordinal categorical variables
                if column == 'employment_length':
                    # Custom encoding for employment length (ordinal)
                    employment_mapping = {
                        '< 1 year': 0, '1-2 years': 1, '3-5 years': 2,
                        '5-10 years': 3, '10+ years': 4, 'Unknown': -1
                    }
                    processed_data[f'{column}_encoded'] = processed_data[column].map(employment_mapping)
                    log['encoding_mappings'][column] = employment_mapping
                    
                else:
                    # Standard label encoding for other categorical variables
                    le = LabelEncoder()
                    # Handle unseen values by adding them to the encoder
                    unique_values = processed_data[column].astype(str).unique()
                    le.fit(unique_values)
                    processed_data[f'{column}_encoded'] = le.transform(processed_data[column].astype(str))
                    
                    self.encoders[column] = le
                    log['encoding_mappings'][column] = dict(zip(le.classes_, le.transform(le.classes_)))
                
                # Drop original categorical column
                processed_data = processed_data.drop(columns=[column])
                log['columns_processed'].append(column)
        
        return processed_data, log
    
    def _engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Create new features from existing ones"""
        processed_data = data.copy()
        log = {'step': 'feature_engineering', 'new_features': []}
        
        # Income-based features
        if 'annual_income' in processed_data.columns and 'loan_amount' in processed_data.columns:
            processed_data['loan_to_income_ratio'] = processed_data['loan_amount'] / processed_data['annual_income']
            log['new_features'].append('loan_to_income_ratio')
        
        if 'monthly_income' not in processed_data.columns and 'annual_income' in processed_data.columns:
            processed_data['monthly_income'] = processed_data['annual_income'] / 12
            log['new_features'].append('monthly_income')
        
        # Credit utilization features
        if 'revol_bal' in processed_data.columns and 'revol_util' in processed_data.columns:
            # Estimate total credit limit
            processed_data['estimated_credit_limit'] = processed_data['revol_bal'] / (processed_data['revol_util'] / 100 + 0.001)
            log['new_features'].append('estimated_credit_limit')
        
        # Age-based features
        if 'age' in processed_data.columns:
            processed_data['age_group'] = pd.cut(
                processed_data['age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(int)
            log['new_features'].append('age_group')
        
        # Credit score bands
        if 'credit_score' in processed_data.columns:
            processed_data['credit_score_band'] = pd.cut(
                processed_data['credit_score'],
                bins=[0, 580, 670, 740, 800, 850],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(int)
            log['new_features'].append('credit_score_band')
        
        # Debt-related features
        if 'debt_to_income_ratio' in processed_data.columns:
            processed_data['high_dti_flag'] = (processed_data['debt_to_income_ratio'] > 0.4).astype(int)
            log['new_features'].append('high_dti_flag')
        
        # Credit history features
        if 'open_acc' in processed_data.columns and 'total_acc' in processed_data.columns:
            processed_data['credit_mix'] = processed_data['open_acc'] / (processed_data['total_acc'] + 1)
            log['new_features'].append('credit_mix')
        
        # Interaction features
        if 'credit_score' in processed_data.columns and 'annual_income' in processed_data.columns:
            # Normalize and create interaction
            credit_norm = (processed_data['credit_score'] - 300) / 550
            income_norm = np.log1p(processed_data['annual_income']) / 15  # Log normalization
            processed_data['credit_income_interaction'] = credit_norm * income_norm
            log['new_features'].append('credit_income_interaction')
        
        return processed_data, log
    
    def _scale_features(self, data: pd.DataFrame, method: str = 'standard', 
                       target_column: str = None) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features"""
        processed_data = data.copy()
        log = {'step': 'feature_scaling', 'method': method, 'columns_scaled': []}
        
        # Get numerical columns (excluding target if specified)
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        if len(numeric_columns) > 0:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
            
            processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
            self.scalers[method] = scaler
            log['columns_scaled'] = numeric_columns
        
        return processed_data, log
    
    def _select_features(self, data: pd.DataFrame, target_column: str, 
                        method: str = 'selectkbest', k: int = 20) -> Tuple[pd.DataFrame, Dict]:
        """Select the most relevant features"""
        if target_column not in data.columns:
            return data, {'step': 'feature_selection', 'error': 'Target column not found'}
        
        processed_data = data.copy()
        X = processed_data.drop(columns=[target_column])
        y = processed_data[target_column]
        
        log = {'step': 'feature_selection', 'method': method, 'features_before': len(X.columns)}
        
        if method == 'selectkbest':
            selector = SelectKBest(score_func=f_classif, k=min(k, len(X.columns)))
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, len(X.columns)))
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
            
        else:
            # Default: keep all features
            selected_features = X.columns.tolist()
        
        # Keep selected features plus target
        final_features = selected_features + [target_column]
        processed_data = processed_data[final_features]
        
        self.feature_selectors[method] = selector if method in ['selectkbest', 'rfe'] else None
        
        log.update({
            'features_after': len(selected_features),
            'selected_features': selected_features,
            'features_removed': len(X.columns) - len(selected_features)
        })
        
        return processed_data, log
    
    def transform_new_data(self, new_data: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        processed_data = new_data.copy()
        
        # Handle missing values
        if 'numeric_simple' in self.imputers:
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                processed_data[numeric_columns] = self.imputers['numeric_simple'].transform(processed_data[numeric_columns])
        
        if 'categorical' in self.imputers:
            categorical_columns = processed_data.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                processed_data[categorical_columns] = self.imputers['categorical'].transform(processed_data[categorical_columns])
        
        # Encode categorical variables
        for column, encoder in self.encoders.items():
            if column in processed_data.columns:
                # Handle unseen categories
                processed_data[column] = processed_data[column].astype(str)
                unseen_mask = ~processed_data[column].isin(encoder.classes_)
                processed_data.loc[unseen_mask, column] = encoder.classes_[0]  # Use first class as default
                
                processed_data[f'{column}_encoded'] = encoder.transform(processed_data[column])
                processed_data = processed_data.drop(columns=[column])
        
        # Apply feature engineering (simplified version)
        processed_data = self._engineer_features(processed_data)[0]
        
        # Scale features
        if 'standard' in self.scalers:
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numeric_columns:
                numeric_columns.remove(target_column)
            
            if len(numeric_columns) > 0:
                processed_data[numeric_columns] = self.scalers['standard'].transform(processed_data[numeric_columns])
        
        return processed_data
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of all preprocessing steps performed"""
        if not self.preprocessing_history:
            return {'message': 'No preprocessing performed yet'}
        
        latest_preprocessing = self.preprocessing_history[-1]
        
        summary = {
            'last_preprocessing': latest_preprocessing['timestamp'],
            'input_shape': latest_preprocessing['input_shape'],
            'output_shape': latest_preprocessing['output_shape'],
            'steps_performed': [step['step'] for step in latest_preprocessing['steps_performed']],
            'total_preprocessing_runs': len(self.preprocessing_history)
        }
        
        # Add details for each step
        for step in latest_preprocessing['steps_performed']:
            step_name = step['step']
            if step_name == 'missing_value_imputation':
                summary[f'{step_name}_details'] = {
                    'strategy': step['strategy'],
                    'columns_processed': len(step['columns_processed'])
                }
            elif step_name == 'outlier_handling':
                summary[f'{step_name}_details'] = {
                    'method': step['method'],
                    'outliers_capped': step['outliers_capped']
                }
            elif step_name == 'categorical_encoding':
                summary[f'{step_name}_details'] = {
                    'columns_encoded': len(step['columns_processed'])
                }
            elif step_name == 'feature_engineering':
                summary[f'{step_name}_details'] = {
                    'new_features_created': len(step['new_features'])
                }
            elif step_name == 'feature_scaling':
                summary[f'{step_name}_details'] = {
                    'method': step['method'],
                    'columns_scaled': len(step['columns_scaled'])
                }
            elif step_name == 'feature_selection':
                summary[f'{step_name}_details'] = {
                    'method': step['method'],
                    'features_selected': step.get('features_after', 0)
                }
        
        return summary
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current datasets"""
        from scipy import stats
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_drift_detected': False,
            'feature_drift_scores': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Get common numeric columns
        ref_numeric = reference_data.select_dtypes(include=[np.number]).columns
        cur_numeric = current_data.select_dtypes(include=[np.number]).columns
        common_numeric = list(set(ref_numeric) & set(cur_numeric))
        
        drift_detected_count = 0
        
        for column in common_numeric:
            ref_values = reference_data[column].dropna()
            cur_values = current_data[column].dropna()
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_values, cur_values)
            
            # Store results
            drift_results['feature_drift_scores'][column] = psi_score
            drift_results['statistical_tests'][column] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi_score': psi_score
            }
            
            # Determine if drift is significant
            drift_threshold = 0.1  # PSI threshold
            if psi_score > drift_threshold or ks_pvalue < 0.05:
                drift_detected_count += 1
        
        # Overall drift assessment
        if drift_detected_count > len(common_numeric) * 0.3:  # If >30% of features show drift
            drift_results['overall_drift_detected'] = True
            drift_results['recommendations'].append("Model retraining recommended due to significant data drift")
        elif drift_detected_count > 0:
            drift_results['recommendations'].append("Monitor model performance - some features show drift")
        else:
            drift_results['recommendations'].append("No significant drift detected")
        
        drift_results['features_with_drift'] = drift_detected_count
        drift_results['total_features_analyzed'] = len(common_numeric)
        
        return drift_results
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        # Create buckets based on reference data
        _, bin_edges = np.histogram(reference, bins=buckets)
        
        # Ensure bins cover the range of both datasets
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if bin_edges[0] > min_val:
            bin_edges[0] = min_val
        if bin_edges[-1] < max_val:
            bin_edges[-1] = max_val
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        cur_props = np.where(cur_props == 0, 0.0001, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi
    
    def generate_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        profile = {
            'timestamp': datetime.now().isoformat(),
            'basic_info': {
                'shape': data.shape,
                'memory_usage': data.memory_usage(deep=True).sum(),
                'duplicate_rows': data.duplicated().sum()
            },
            'column_info': {},
            'data_quality': {
                'missing_data_percentage': (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
                'complete_rows': data.dropna().shape[0],
                'completion_rate': (data.dropna().shape[0] / data.shape[0]) * 100
            },
            'correlations': {}
        }
        
        # Analyze each column
        for column in data.columns:
            col_info = {
                'dtype': str(data[column].dtype),
                'missing_count': data[column].isnull().sum(),
                'missing_percentage': (data[column].isnull().sum() / len(data)) * 100,
                'unique_values': data[column].nunique(),
                'unique_percentage': (data[column].nunique() / len(data)) * 100
            }
            
            if data[column].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'median': data[column].median(),
                    'skewness': data[column].skew(),
                    'kurtosis': data[column].kurtosis()
                })
            else:
                value_counts = data[column].value_counts()
                col_info.update({
                    'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'top_value_frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'top_value_percentage': (value_counts.iloc[0] / len(data)) * 100 if len(value_counts) > 0 else 0
                })
            
            profile['column_info'][column] = col_info
        
        # Calculate correlations for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            
            # Find high correlations (>0.8)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            profile['correlations'] = {
                'high_correlation_pairs': high_corr_pairs,
                'correlation_matrix': corr_matrix.to_dict()
            }
        
        return profile
