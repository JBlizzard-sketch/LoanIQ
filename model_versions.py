import os
import json
import pickle
import shutil
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional

class ModelVersionManager:
    def __init__(self, base_path='model_versions'):
        self.base_path = base_path
        self.versions_file = os.path.join(base_path, 'versions.json')
        self.models_dir = os.path.join(base_path, 'models')
        self.metadata_dir = os.path.join(base_path, 'metadata')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize versions tracking file
        self._initialize_versions_file()
    
    def _initialize_versions_file(self):
        """Initialize the versions tracking file"""
        if not os.path.exists(self.versions_file):
            initial_data = {
                'latest_version': '0.0.0',
                'versions': []
            }
            with open(self.versions_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def _load_versions_data(self):
        """Load versions data from file"""
        try:
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading versions data: {e}")
            return {'latest_version': '0.0.0', 'versions': []}
    
    def _save_versions_data(self, data):
        """Save versions data to file"""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving versions data: {e}")
    
    def _increment_version(self, current_version, version_type='patch'):
        """Increment version number based on type (major, minor, patch)"""
        try:
            major, minor, patch = map(int, current_version.split('.'))
            
            if version_type == 'major':
                major += 1
                minor = 0
                patch = 0
            elif version_type == 'minor':
                minor += 1
                patch = 0
            else:  # patch
                patch += 1
            
            return f"{major}.{minor}.{patch}"
        except ValueError:
            return "1.0.0"
    
    def save_model_version(self, model, model_name: str, metrics: Dict, 
                          training_data_info: Dict, version_type='patch',
                          description: str = "", tags: List[str] = None):
        """Save a new model version"""
        
        versions_data = self._load_versions_data()
        current_version = versions_data['latest_version']
        new_version = self._increment_version(current_version, version_type)
        
        # Generate version ID
        version_id = f"{model_name}_{new_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create version directory
        version_dir = os.path.join(self.models_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, 'model.pkl')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
        
        # Create metadata
        metadata = {
            'version_id': version_id,
            'model_name': model_name,
            'version': new_version,
            'created_date': datetime.now().isoformat(),
            'description': description,
            'tags': tags or [],
            'metrics': metrics,
            'training_data_info': training_data_info,
            'model_path': model_path,
            'status': 'active',
            'model_size': self._get_file_size(model_path),
            'training_time': training_data_info.get('training_time', 'Unknown'),
            'performance': {
                'auc': metrics.get('auc_score', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, f"{version_id}_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return None
        
        # Update versions tracking
        versions_data['latest_version'] = new_version
        versions_data['versions'].append(metadata)
        self._save_versions_data(versions_data)
        
        print(f"Model version {new_version} saved successfully with ID: {version_id}")
        return version_id
    
    def load_model_version(self, version_id: str):
        """Load a specific model version"""
        version_dir = os.path.join(self.models_dir, version_id)
        model_path = os.path.join(version_dir, 'model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model version {version_id} not found")
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(self.metadata_dir, f"{version_id}_metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return {'model': model, 'metadata': metadata}
        
        except Exception as e:
            print(f"Error loading model version {version_id}: {e}")
            return None
    
    def get_all_versions(self) -> List[Dict]:
        """Get all model versions"""
        versions_data = self._load_versions_data()
        return versions_data.get('versions', [])
    
    def get_versions_by_model(self, model_name: str) -> List[Dict]:
        """Get all versions for a specific model"""
        all_versions = self.get_all_versions()
        return [v for v in all_versions if v['model_name'] == model_name]
    
    def get_latest_version(self, model_name: str = None) -> Optional[Dict]:
        """Get the latest version for a model or overall"""
        all_versions = self.get_all_versions()
        
        if model_name:
            model_versions = [v for v in all_versions if v['model_name'] == model_name]
            if not model_versions:
                return None
            return max(model_versions, key=lambda x: x['created_date'])
        else:
            if not all_versions:
                return None
            return max(all_versions, key=lambda x: x['created_date'])
    
    def delete_model_version(self, version_id: str):
        """Delete a specific model version"""
        # Remove model files
        version_dir = os.path.join(self.models_dir, version_id)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        
        # Remove metadata file
        metadata_path = os.path.join(self.metadata_dir, f"{version_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Update versions tracking
        versions_data = self._load_versions_data()
        versions_data['versions'] = [v for v in versions_data['versions'] if v['version_id'] != version_id]
        self._save_versions_data(versions_data)
        
        print(f"Model version {version_id} deleted successfully")
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict:
        """Compare two model versions"""
        versions_data = self._load_versions_data()
        
        version1 = next((v for v in versions_data['versions'] if v['version_id'] == version_id1), None)
        version2 = next((v for v in versions_data['versions'] if v['version_id'] == version_id2), None)
        
        if not version1 or not version2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': {
                'id': version_id1,
                'version': version1['version'],
                'model_name': version1['model_name'],
                'created_date': version1['created_date'],
                'metrics': version1['metrics']
            },
            'version2': {
                'id': version_id2,
                'version': version2['version'],
                'model_name': version2['model_name'],
                'created_date': version2['created_date'],
                'metrics': version2['metrics']
            },
            'differences': {}
        }
        
        # Compare metrics
        for metric in ['auc_score', 'accuracy', 'precision', 'recall', 'f1_score']:
            if metric in version1['metrics'] and metric in version2['metrics']:
                diff = version2['metrics'][metric] - version1['metrics'][metric]
                comparison['differences'][metric] = {
                    'difference': diff,
                    'percentage_change': (diff / version1['metrics'][metric]) * 100 if version1['metrics'][metric] != 0 else 0,
                    'better': 'version2' if diff > 0 else 'version1' if diff < 0 else 'equal'
                }
        
        return comparison
    
    def set_version_status(self, version_id: str, status: str):
        """Set the status of a model version (active, archived, deprecated)"""
        versions_data = self._load_versions_data()
        
        for version in versions_data['versions']:
            if version['version_id'] == version_id:
                version['status'] = status
                break
        
        self._save_versions_data(versions_data)
        
        # Update metadata file
        metadata_path = os.path.join(self.metadata_dir, f"{version_id}_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['status'] = status
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"Error updating metadata status: {e}")
    
    def get_version_performance_history(self, model_name: str) -> pd.DataFrame:
        """Get performance history for a model across versions"""
        model_versions = self.get_versions_by_model(model_name)
        
        if not model_versions:
            return pd.DataFrame()
        
        performance_data = []
        for version in model_versions:
            perf_data = {
                'version_id': version['version_id'],
                'version': version['version'],
                'created_date': version['created_date'],
                'status': version['status']
            }
            
            # Add metrics
            for metric, value in version['metrics'].items():
                perf_data[metric] = value
            
            performance_data.append(perf_data)
        
        df = pd.DataFrame(performance_data)
        df['created_date'] = pd.to_datetime(df['created_date'])
        df = df.sort_values('created_date')
        
        return df
    
    def export_version(self, version_id: str, export_path: str):
        """Export a model version to a specified path"""
        version_dir = os.path.join(self.models_dir, version_id)
        metadata_path = os.path.join(self.metadata_dir, f"{version_id}_metadata.json")
        
        if not os.path.exists(version_dir):
            raise FileNotFoundError(f"Model version {version_id} not found")
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Copy model files
        export_model_dir = os.path.join(export_path, version_id)
        shutil.copytree(version_dir, export_model_dir)
        
        # Copy metadata
        if os.path.exists(metadata_path):
            shutil.copy2(metadata_path, os.path.join(export_path, f"{version_id}_metadata.json"))
        
        print(f"Model version {version_id} exported to {export_path}")
    
    def import_version(self, import_path: str, version_id: str):
        """Import a model version from a specified path"""
        source_model_dir = os.path.join(import_path, version_id)
        source_metadata = os.path.join(import_path, f"{version_id}_metadata.json")
        
        if not os.path.exists(source_model_dir):
            raise FileNotFoundError(f"Source model directory not found: {source_model_dir}")
        
        # Copy model files
        target_model_dir = os.path.join(self.models_dir, version_id)
        shutil.copytree(source_model_dir, target_model_dir)
        
        # Copy metadata
        if os.path.exists(source_metadata):
            target_metadata = os.path.join(self.metadata_dir, f"{version_id}_metadata.json")
            shutil.copy2(source_metadata, target_metadata)
            
            # Load metadata and update versions tracking
            try:
                with open(target_metadata, 'r') as f:
                    metadata = json.load(f)
                
                versions_data = self._load_versions_data()
                versions_data['versions'].append(metadata)
                self._save_versions_data(versions_data)
                
            except Exception as e:
                print(f"Error updating versions tracking: {e}")
        
        print(f"Model version {version_id} imported successfully")
    
    def cleanup_old_versions(self, model_name: str, keep_latest: int = 5):
        """Clean up old model versions, keeping only the latest N versions"""
        model_versions = self.get_versions_by_model(model_name)
        
        if len(model_versions) <= keep_latest:
            return
        
        # Sort by creation date (newest first)
        model_versions.sort(key=lambda x: x['created_date'], reverse=True)
        
        # Delete older versions
        versions_to_delete = model_versions[keep_latest:]
        
        for version in versions_to_delete:
            if version['status'] != 'active':  # Don't delete active versions
                self.delete_model_version(version['version_id'])
                print(f"Deleted old version: {version['version_id']}")
    
    def get_model_statistics(self) -> Dict:
        """Get statistics about all model versions"""
        all_versions = self.get_all_versions()
        
        if not all_versions:
            return {}
        
        stats = {
            'total_versions': len(all_versions),
            'models': {},
            'status_distribution': {},
            'performance_summary': {}
        }
        
        # Model distribution
        model_counts = {}
        for version in all_versions:
            model_name = version['model_name']
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        stats['models'] = model_counts
        
        # Status distribution
        status_counts = {}
        for version in all_versions:
            status = version['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        stats['status_distribution'] = status_counts
        
        # Performance summary
        metrics = ['auc_score', 'accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            values = [v['metrics'].get(metric, 0) for v in all_versions if metric in v['metrics']]
            if values:
                stats['performance_summary'][metric] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return stats
    
    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size"""
        try:
            size_bytes = os.path.getsize(file_path)
            
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        except:
            return "Unknown"
    
    def create_model_deployment_package(self, version_id: str, output_path: str):
        """Create a deployment package for a model version"""
        version_data = self.load_model_version(version_id)
        
        if not version_data:
            raise ValueError(f"Version {version_id} not found")
        
        # Create deployment package directory
        package_dir = os.path.join(output_path, f"deployment_{version_id}")
        os.makedirs(package_dir, exist_ok=True)
        
        # Copy model
        model_path = os.path.join(package_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(version_data['model'], f)
        
        # Create deployment configuration
        deployment_config = {
            'version_id': version_id,
            'model_info': version_data['metadata'],
            'deployment_date': datetime.now().isoformat(),
            'requirements': [
                'pandas>=1.3.0',
                'numpy>=1.21.0',
                'scikit-learn>=1.0.0',
                'xgboost>=1.5.0',
                'lightgbm>=3.3.0'
            ]
        }
        
        config_path = os.path.join(package_dir, 'deployment_config.json')
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        # Create deployment script template
        deployment_script = f'''
import pickle
import json
import pandas as pd
import numpy as np

class ModelDeployment:
    def __init__(self):
        # Load model
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load configuration
        with open('deployment_config.json', 'r') as f:
            self.config = json.load(f)
    
    def predict(self, input_data):
        """Make prediction on input data"""
        # Implement prediction logic here
        # This is a template - customize based on your preprocessing needs
        
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        
        # Preprocess data (customize as needed)
        # df_processed = self.preprocess_data(df)
        
        # Make prediction
        prediction = self.model.predict_proba(df)[:, 1]
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def get_model_info(self):
        """Get model information"""
        return self.config['model_info']

# Example usage:
# deployment = ModelDeployment()
# result = deployment.predict({{'feature1': value1, 'feature2': value2}})
'''
        
        script_path = os.path.join(package_dir, 'deployment.py')
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        
        print(f"Deployment package created at: {package_dir}")
        return package_dir
