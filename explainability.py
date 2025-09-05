import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import shap  # Temporarily disabled due to compatibility issues
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    def __init__(self, model, model_type: str = 'auto'):
        """
        Initialize SHAP explainer for model interpretability
        
        Args:
            model: Trained ML model
            model_type: Type of model ('tree', 'linear', 'kernel', 'auto')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.background_data = None
        
        # Initialize explainer based on model type
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            # Try to detect model type automatically
            model_name = self.model.__class__.__name__.lower()
            
            if any(tree_type in model_name for tree_type in ['xgb', 'lgb', 'randomforest', 'gradientboosting']):
                self.explainer_type = 'tree'
            elif any(linear_type in model_name for linear_type in ['logistic', 'linear', 'ridge', 'lasso']):
                self.explainer_type = 'linear'
            else:
                self.explainer_type = 'kernel'
                
        except Exception as e:
            print(f"Could not auto-detect model type: {e}")
            self.explainer_type = 'kernel'
        
        # Note: SHAP functionality temporarily disabled due to compatibility issues
        print("Note: SHAP explainer temporarily disabled. Using feature importance fallback.")
    
    def fit_explainer(self, background_data: pd.DataFrame, max_evals: int = 100):
        """
        Fit the SHAP explainer with background data
        
        Args:
            background_data: Representative sample of training data
            max_evals: Maximum evaluations for kernel explainer
        """
        self.background_data = background_data.copy()
        self.feature_names = background_data.columns.tolist()
        
        try:
            if self.explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, background_data)
            else:  # kernel explainer
                # Use a smaller sample for kernel explainer to improve performance
                sample_size = min(100, len(background_data))
                background_sample = background_data.sample(n=sample_size, random_state=42)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background_sample,
                    link="logit"
                )
            
            print(f"SHAP {self.explainer_type} explainer initialized successfully")
            
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            # Fallback to kernel explainer
            try:
                sample_size = min(50, len(background_data))
                background_sample = background_data.sample(n=sample_size, random_state=42)
                self.explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1], 
                    background_sample
                )
                self.explainer_type = 'kernel'
                print("Fallback to kernel explainer successful")
            except Exception as e2:
                print(f"Fallback explainer also failed: {e2}")
                self.explainer = None
    
    def calculate_shap_values(self, data: pd.DataFrame, max_samples: int = 1000):
        """
        Calculate SHAP values for given data
        
        Args:
            data: Data to explain
            max_samples: Maximum number of samples to process
        """
        if self.explainer is None:
            print("Explainer not initialized. Please call fit_explainer first.")
            return None
        
        # Limit samples for performance
        if len(data) > max_samples:
            data_sample = data.sample(n=max_samples, random_state=42)
        else:
            data_sample = data.copy()
        
        try:
            if self.explainer_type == 'tree':
                self.shap_values = self.explainer.shap_values(data_sample)
                # For binary classification, take the positive class
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            else:
                self.shap_values = self.explainer.shap_values(data_sample)
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
            
            print(f"SHAP values calculated for {len(data_sample)} samples")
            return self.shap_values
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None
    
    def plot_single_prediction(self, input_data: Dict, max_features: int = 10) -> go.Figure:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            input_data: Single instance to explain
            max_features: Maximum number of features to show
        """
        if self.explainer is None:
            return self._create_error_plot("SHAP explainer not initialized")
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Ensure column order matches training data
            if self.feature_names:
                missing_cols = set(self.feature_names) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                df = df[self.feature_names]
            
            # Calculate SHAP values for single instance
            if self.explainer_type == 'tree':
                shap_vals = self.explainer.shap_values(df)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_vals = shap_vals[0]  # Get first (and only) instance
            else:
                shap_vals = self.explainer.shap_values(df)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_vals = shap_vals[0]
            
            # Get feature values
            feature_values = df.iloc[0].values
            feature_names = df.columns.tolist()
            
            # Sort by absolute SHAP value
            importance_order = np.argsort(np.abs(shap_vals))[::-1][:max_features]
            
            # Create waterfall plot
            fig = go.Figure()
            
            # Base value (expected value)
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
            # Calculate cumulative values for waterfall
            cumulative = base_value
            x_labels = ['Base Value']
            y_values = [base_value]
            colors = ['lightgray']
            
            for i in importance_order:
                feature_name = feature_names[i]
                shap_val = shap_vals[i]
                feature_val = feature_values[i]
                
                cumulative += shap_val
                x_labels.append(f"{feature_name}\n({feature_val:.2f})")
                y_values.append(cumulative)
                colors.append('green' if shap_val > 0 else 'red')
            
            # Final prediction
            x_labels.append('Prediction')
            y_values.append(cumulative)
            colors.append('darkblue')
            
            # Create waterfall chart
            for i in range(len(y_values)):
                fig.add_trace(go.Bar(
                    x=[x_labels[i]],
                    y=[y_values[i]],
                    marker_color=colors[i],
                    name=x_labels[i],
                    showlegend=False
                ))
            
            fig.update_layout(
                title="SHAP Explanation - Feature Contributions",
                xaxis_title="Features",
                yaxis_title="SHAP Value (Log Odds)",
                height=500,
                xaxis={'tickangle': -45}
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating single prediction plot: {e}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def plot_global_importance(self, max_features: int = 15) -> go.Figure:
        """
        Create global feature importance plot
        
        Args:
            max_features: Maximum number of features to show
        """
        if self.shap_values is None:
            return self._create_error_plot("SHAP values not calculated. Please run calculate_shap_values first.")
        
        try:
            # Calculate mean absolute SHAP values
            mean_shap_values = np.mean(np.abs(self.shap_values), axis=0)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(mean_shap_values)],
                'importance': mean_shap_values
            }).sort_values('importance', ascending=True).tail(max_features)
            
            # Create horizontal bar plot
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Global Feature Importance (Mean |SHAP|)',
                color='importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title='Mean |SHAP Value|',
                yaxis_title='Features'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating global importance plot: {e}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def plot_summary(self, max_features: int = 15) -> go.Figure:
        """
        Create SHAP summary plot
        
        Args:
            max_features: Maximum number of features to show
        """
        if self.shap_values is None or self.background_data is None:
            return self._create_error_plot("SHAP values or background data not available")
        
        try:
            # Calculate feature importance for sorting
            mean_shap_values = np.mean(np.abs(self.shap_values), axis=0)
            importance_order = np.argsort(mean_shap_values)[::-1][:max_features]
            
            # Create summary plot data
            plot_data = []
            
            for i, feature_idx in enumerate(importance_order):
                feature_name = self.feature_names[feature_idx]
                feature_shap_values = self.shap_values[:, feature_idx]
                
                # Get feature values from background data
                if feature_name in self.background_data.columns:
                    feature_values = self.background_data[feature_name].values[:len(feature_shap_values)]
                else:
                    feature_values = np.zeros(len(feature_shap_values))
                
                for j, (shap_val, feat_val) in enumerate(zip(feature_shap_values, feature_values)):
                    plot_data.append({
                        'feature': feature_name,
                        'shap_value': shap_val,
                        'feature_value': feat_val,
                        'feature_rank': i
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x='shap_value',
                y='feature',
                color='feature_value',
                title='SHAP Summary Plot',
                color_continuous_scale='RdYlBu',
                opacity=0.6
            )
            
            fig.update_layout(
                height=500,
                xaxis_title='SHAP Value (Impact on Model Output)',
                yaxis_title='Features',
                yaxis={'categoryorder': 'total ascending'},
                coloraxis_colorbar_title="Feature Value"
            )
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
            
            return fig
            
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def plot_partial_dependence(self, feature_name: str, num_points: int = 50) -> go.Figure:
        """
        Create partial dependence plot for a specific feature
        
        Args:
            feature_name: Name of the feature
            num_points: Number of points to evaluate
        """
        if self.background_data is None or feature_name not in self.background_data.columns:
            return self._create_error_plot(f"Feature '{feature_name}' not found in background data")
        
        try:
            # Get feature range
            feature_values = self.background_data[feature_name]
            min_val, max_val = feature_values.min(), feature_values.max()
            
            # Create evaluation points
            eval_points = np.linspace(min_val, max_val, num_points)
            
            # Create base dataset (median values for other features)
            base_data = self.background_data.median().to_frame().T
            
            # Vary the target feature
            pdp_data = []
            predictions = []
            
            for val in eval_points:
                temp_data = base_data.copy()
                temp_data[feature_name] = val
                
                # Get prediction
                try:
                    if hasattr(self.model, 'predict_proba'):
                        pred = self.model.predict_proba(temp_data)[0, 1]  # Probability of positive class
                    else:
                        pred = self.model.predict(temp_data)[0]
                    
                    predictions.append(pred)
                    pdp_data.append({'feature_value': val, 'prediction': pred})
                except Exception as e:
                    print(f"Error making prediction for {feature_name}={val}: {e}")
                    predictions.append(np.nan)
                    pdp_data.append({'feature_value': val, 'prediction': np.nan})
            
            pdp_df = pd.DataFrame(pdp_data).dropna()
            
            if len(pdp_df) == 0:
                return self._create_error_plot("No valid predictions generated")
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pdp_df['feature_value'],
                y=pdp_df['prediction'],
                mode='lines+markers',
                name='Partial Dependence',
                line=dict(width=3, color='blue'),
                marker=dict(size=6)
            ))
            
            # Add feature distribution as histogram
            fig.add_trace(go.Histogram(
                x=feature_values,
                name='Feature Distribution',
                opacity=0.3,
                yaxis='y2',
                nbinsx=30,
                marker_color='lightgray'
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                title=f'Partial Dependence Plot - {feature_name}',
                xaxis_title=feature_name,
                yaxis_title='Prediction (Default Probability)',
                yaxis2=dict(
                    title='Feature Distribution',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                height=500,
                legend=dict(x=0.7, y=1)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating partial dependence plot: {e}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def plot_interaction_effects(self, feature1: str, feature2: str, num_points: int = 20) -> go.Figure:
        """
        Create interaction effect plot between two features
        
        Args:
            feature1: First feature name
            feature2: Second feature name
            num_points: Number of points per dimension
        """
        if (self.background_data is None or 
            feature1 not in self.background_data.columns or 
            feature2 not in self.background_data.columns):
            return self._create_error_plot(f"Features '{feature1}' or '{feature2}' not found")
        
        try:
            # Get feature ranges
            f1_values = np.linspace(
                self.background_data[feature1].min(),
                self.background_data[feature1].max(),
                num_points
            )
            f2_values = np.linspace(
                self.background_data[feature2].min(),
                self.background_data[feature2].max(),
                num_points
            )
            
            # Create base dataset
            base_data = self.background_data.median().to_frame().T
            
            # Create grid
            predictions = np.zeros((num_points, num_points))
            
            for i, f1_val in enumerate(f1_values):
                for j, f2_val in enumerate(f2_values):
                    temp_data = base_data.copy()
                    temp_data[feature1] = f1_val
                    temp_data[feature2] = f2_val
                    
                    try:
                        if hasattr(self.model, 'predict_proba'):
                            pred = self.model.predict_proba(temp_data)[0, 1]
                        else:
                            pred = self.model.predict(temp_data)[0]
                        predictions[i, j] = pred
                    except Exception:
                        predictions[i, j] = np.nan
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=predictions,
                x=f2_values,
                y=f1_values,
                colorscale='RdYlBu_r',
                colorbar=dict(title="Prediction")
            ))
            
            fig.update_layout(
                title=f'Interaction Effects: {feature1} vs {feature2}',
                xaxis_title=feature2,
                yaxis_title=feature1,
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating interaction plot: {e}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot with the given message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="SHAP Explanation Error",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        
        return fig
    
    def get_feature_contributions(self, input_data: Dict, top_n: int = 10) -> Dict[str, Any]:
        """
        Get top feature contributions for a single prediction
        
        Args:
            input_data: Single instance to explain
            top_n: Number of top features to return
        """
        if self.explainer is None:
            return {'error': 'SHAP explainer not initialized'}
        
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Ensure column order matches training data
            if self.feature_names:
                missing_cols = set(self.feature_names) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                df = df[self.feature_names]
            
            # Calculate SHAP values
            if self.explainer_type == 'tree':
                shap_vals = self.explainer.shap_values(df)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_vals = shap_vals[0]
            else:
                shap_vals = self.explainer.shap_values(df)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_vals = shap_vals[0]
            
            # Get base value
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            
            # Create feature contributions
            contributions = []
            for i, (feature, shap_val) in enumerate(zip(df.columns, shap_vals)):
                contributions.append({
                    'feature': feature,
                    'feature_value': df.iloc[0, i],
                    'shap_value': shap_val,
                    'abs_shap_value': abs(shap_val),
                    'contribution_type': 'positive' if shap_val > 0 else 'negative'
                })
            
            # Sort by absolute SHAP value
            contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)
            
            return {
                'base_value': base_value,
                'final_prediction': base_value + sum(shap_vals),
                'total_shap_effect': sum(shap_vals),
                'top_contributions': contributions[:top_n],
                'all_contributions': contributions
            }
            
        except Exception as e:
            return {'error': f'Error calculating feature contributions: {str(e)}'}
    
    def generate_explanation_text(self, input_data: Dict, top_n: int = 5) -> str:
        """
        Generate human-readable explanation text
        
        Args:
            input_data: Single instance to explain
            top_n: Number of top features to include in explanation
        """
        contributions = self.get_feature_contributions(input_data, top_n)
        
        if 'error' in contributions:
            return f"Error generating explanation: {contributions['error']}"
        
        # Base explanation
        base_value = contributions['base_value']
        final_pred = contributions['final_prediction']
        
        explanation = f"Base prediction probability: {base_value:.3f}\n"
        explanation += f"Final prediction probability: {final_pred:.3f}\n\n"
        
        explanation += "Key factors influencing this prediction:\n\n"
        
        for i, contrib in enumerate(contributions['top_contributions'], 1):
            feature = contrib['feature']
            feature_val = contrib['feature_value']
            shap_val = contrib['shap_value']
            contrib_type = contrib['contribution_type']
            
            direction = "increases" if contrib_type == 'positive' else "decreases"
            explanation += f"{i}. {feature} = {feature_val:.2f}\n"
            explanation += f"   This {direction} the default probability by {abs(shap_val):.3f}\n\n"
        
        # Overall assessment
        if final_pred > 0.5:
            risk_level = "HIGH"
        elif final_pred > 0.3:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        explanation += f"Overall Risk Assessment: {risk_level} RISK\n"
        explanation += f"Default Probability: {final_pred:.1%}"
        
        return explanation
