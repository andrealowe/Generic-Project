"""
Quick Model Training for Diesel Fuel Quality Prediction Demo

This simplified version trains models quickly for demonstration purposes.
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

class QuickModelTrainer:
    """Quick model trainer for demo purposes."""
    
    def __init__(self, random_state=42):
        """Initialize the trainer."""
        self.random_state = random_state
        self.paths = get_data_paths()
        self.target_columns = [
            'cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 
            'density_gcc', 'cloud_point_c', 'aromatic_content', 'quality_score'
        ]
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset."""
        print("Loading dataset...")
        
        dataset_path = self.paths['processed_data_path'] / 'diesel_quality_dataset.csv'
        df = pd.read_csv(dataset_path)
        
        # Select feature columns
        exclude_cols = (
            self.target_columns + 
            ['timestamp', 'quality_pass', 'is_anomaly', 'month', 
             'cetane_pass', 'sulfur_pass', 'viscosity_pass', 'flash_point_pass',
             'cloud_point_pass', 'aromatic_pass', 'pour_point_c', 'distillation_t90']
        )
        
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].fillna(df[feature_columns].median())
        y = df[self.target_columns].fillna(df[self.target_columns].median())
        
        print(f"Features: {X.shape[1]}, Targets: {y.shape[1]}, Samples: {X.shape[0]}")
        
        return X, y, feature_columns
    
    def train_models_for_target(self, X, y, target, feature_columns):
        """Train models for a specific target."""
        print(f"Training models for: {target}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        metrics = {}
        
        # Random Forest
        print(f"  Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        
        rf_metrics = {
            'r2': r2_score(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'mae': mean_absolute_error(y_test, rf_pred)
        }
        
        models['random_forest'] = rf
        metrics['random_forest'] = rf_metrics
        
        print(f"    R² = {rf_metrics['r2']:.4f}, RMSE = {rf_metrics['rmse']:.4f}")
        
        # XGBoost
        print(f"  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=50,  # Reduced for speed
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        xgb_metrics = {
            'r2': r2_score(y_test, xgb_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'mae': mean_absolute_error(y_test, xgb_pred)
        }
        
        models['xgboost'] = xgb_model
        metrics['xgboost'] = xgb_metrics
        
        print(f"    R² = {xgb_metrics['r2']:.4f}, RMSE = {xgb_metrics['rmse']:.4f}")
        
        # Select best model
        best_model_name = 'random_forest' if rf_metrics['r2'] > xgb_metrics['r2'] else 'xgboost'
        best_model = models[best_model_name]
        best_metrics = metrics[best_model_name]
        
        print(f"  Best model: {best_model_name} (R² = {best_metrics['r2']:.4f})")
        
        return {
            'model': best_model,
            'model_name': best_model_name,
            'scaler': scaler,
            'metrics': best_metrics,
            'all_metrics': metrics,
            'feature_columns': feature_columns
        }
    
    def train_all_models(self):
        """Train models for all targets."""
        print("QUICK DIESEL FUEL QUALITY MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        X, y, feature_columns = self.load_and_prepare_data()
        
        # Train models for each target
        best_models = {}
        all_results = {}
        
        for target in self.target_columns:
            print(f"\n{'-'*50}")
            result = self.train_models_for_target(X, y[target], target, feature_columns)
            
            best_models[target] = {
                'model': result['model'],
                'model_name': result['model_name'],
                'r2_score': result['metrics']['r2'],
                'rmse': result['metrics']['rmse']
            }
            
            self.models[target] = result['model']
            self.scalers[target] = result['scaler']
            all_results[target] = result['all_metrics']
        
        # Save models and artifacts
        self.save_models_and_artifacts(best_models, all_results, feature_columns)
        
        # Generate report
        self.generate_training_report(best_models, all_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("MODEL TRAINING COMPLETE")
        print(f"{'='*60}")
        
        avg_r2 = np.mean([info['r2_score'] for info in best_models.values()])
        print(f"Models trained: {len(best_models)}")
        print(f"Average R² score: {avg_r2:.4f}")
        print(f"Best performing targets:")
        
        sorted_models = sorted(best_models.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        for target, info in sorted_models[:3]:
            print(f"  {target}: {info['model_name']} (R² = {info['r2_score']:.4f})")
        
        return best_models, all_results
    
    def save_models_and_artifacts(self, best_models, all_results, feature_columns):
        """Save models and artifacts."""
        print("\nSaving models and artifacts...")
        
        models_dir = self.paths['models_path']
        
        for target, model_info in best_models.items():
            # Save best model
            model_path = models_dir / f"best_{target}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            
            # Save scaler
            scaler_path = models_dir / f"scaler_{target}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[target], f)
        
        # Save feature columns
        feature_path = models_dir / "feature_columns.json"
        with open(feature_path, 'w') as f:
            json.dump(feature_columns, f)
        
        # Save performance summary
        performance_path = models_dir / "model_performance.json"
        performance_summary = {
            'best_models': {
                target: {
                    'model_name': info['model_name'],
                    'r2_score': info['r2_score'],
                    'rmse': info['rmse']
                }
                for target, info in best_models.items()
            },
            'all_results': all_results,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(feature_columns)
        }
        
        with open(performance_path, 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        print(f"Models saved to: {models_dir}")
    
    def generate_training_report(self, best_models, all_results):
        """Generate training report."""
        print("\nGenerating training report...")
        
        report_path = self.paths['reports_path'] / 'model_training_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Diesel Fuel Quality Model Training Report\n\n")
            f.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            avg_r2 = np.mean([info['r2_score'] for info in best_models.values()])
            f.write(f"- **Average R² Score:** {avg_r2:.4f}\n")
            f.write(f"- **Models Trained:** {len(best_models)} quality parameters\n")
            f.write("- **Algorithms Used:** Random Forest, XGBoost\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Target Parameter | Best Model | R² Score | RMSE |\n")
            f.write("|-----------------|------------|----------|------|\n")
            
            for target, info in best_models.items():
                f.write(f"| {target} | {info['model_name']} | {info['r2_score']:.4f} | {info['rmse']:.4f} |\n")
            
            f.write("\n## Model Comparison\n\n")
            for target in self.target_columns:
                if target in all_results:
                    f.write(f"### {target.replace('_', ' ').title()}\n\n")
                    f.write("| Model | R² Score | RMSE |\n")
                    f.write("|-------|----------|------|\n")
                    
                    for model_name, metrics in all_results[target].items():
                        f.write(f"| {model_name} | {metrics['r2']:.4f} | {metrics['rmse']:.4f} |\n")
                    f.write("\n")
            
            f.write("## Production Readiness\n\n")
            
            high_perf = [t for t, info in best_models.items() if info['r2_score'] > 0.85]
            medium_perf = [t for t, info in best_models.items() if 0.7 <= info['r2_score'] <= 0.85]
            
            if high_perf:
                f.write("**High Performance Models (R² > 0.85):**\n")
                for target in high_perf:
                    f.write(f"- {target}: Production ready\n")
                f.write("\n")
            
            if medium_perf:
                f.write("**Medium Performance Models (0.7 ≤ R² ≤ 0.85):**\n")
                for target in medium_perf:
                    f.write(f"- {target}: Consider additional tuning\n")
                f.write("\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Deploy models to prediction API\n")
            f.write("2. Create monitoring dashboard\n")
            f.write("3. Set up automated retraining pipeline\n")
            f.write("4. Implement prediction confidence intervals\n")
        
        print(f"Training report saved to: {report_path}")

def main():
    """Main function."""
    trainer = QuickModelTrainer(random_state=42)
    best_models, all_results = trainer.train_all_models()
    return trainer, best_models, all_results

if __name__ == "__main__":
    trainer, best_models, all_results = main()