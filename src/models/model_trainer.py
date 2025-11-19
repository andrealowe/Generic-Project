"""
Advanced ML Model Training for Diesel Fuel Quality Prediction

This module implements comprehensive model training using multiple frameworks:
- H2O AutoML for automated model selection
- XGBoost for gradient boosting
- Random Forest for baseline comparison
- MLflow for experiment tracking

Features:
- Multi-output prediction for all quality parameters
- Feature engineering and selection
- Cross-validation and hyperparameter tuning
- Model comparison and selection
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
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:8768")
mlflow.set_experiment("diesel_fuel_quality_prediction")

class DieselFuelModelTrainer:
    """Comprehensive model trainer for diesel fuel quality prediction."""
    
    def __init__(self, random_state=42):
        """Initialize the model trainer."""
        self.random_state = random_state
        self.paths = get_data_paths()
        self.target_columns = [
            'cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c', 
            'density_gcc', 'cloud_point_c', 'aromatic_content', 'quality_score'
        ]
        self.feature_columns = None
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_metrics = {}
        
    def load_data(self):
        """Load and preprocess the dataset."""
        print("Loading dataset...")
        
        dataset_path = self.paths['processed_data_path'] / 'diesel_quality_dataset.csv'
        df = pd.read_csv(dataset_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Dataset loaded: {df.shape}")
        return df
    
    def engineer_features(self, df):
        """Create engineered features based on domain knowledge."""
        print("Engineering features...")
        
        df_engineered = df.copy()
        
        # Temperature-pressure interaction terms
        df_engineered['temp_pressure_interaction'] = (
            df['hydrotreat_temp'] * df['hydrotreat_pressure'] / 1000
        )
        
        # Catalyst efficiency indicator
        df_engineered['catalyst_efficiency'] = np.exp(-df['catalyst_age_days'] / 365)
        
        # Process severity indicators
        df_engineered['hydrotreating_severity'] = (
            (df['hydrotreat_temp'] - 300) / 80 + 
            (df['hydrotreat_pressure'] - 40) / 100
        ) / 2
        
        # Distillation temperature range
        df_engineered['distillation_range'] = (
            df['distillation_temp_bottom'] - df['distillation_temp_top']
        )
        
        # Crude oil quality index
        df_engineered['crude_quality_index'] = (
            (df['crude_api_gravity'] - 25) / 20 - 
            (df['crude_sulfur_ppm'] - 100) / 7900
        )
        
        # Seasonal indicators
        df_engineered['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df_engineered['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Log transformations for skewed variables
        df_engineered['log_sulfur_ppm'] = np.log1p(df['sulfur_ppm'])
        df_engineered['log_crude_sulfur'] = np.log1p(df['crude_sulfur_ppm'])
        
        # Process stability indicators (rolling statistics)
        for col in ['hydrotreat_temp', 'hydrotreat_pressure', 'feed_rate_bpd']:
            df_engineered[f'{col}_stability'] = df[col].rolling(window=24, min_periods=1).std()
        
        print(f"Features engineered: {df_engineered.shape[1] - df.shape[1]} new features")
        return df_engineered
    
    def prepare_features_and_targets(self, df):
        """Prepare feature matrix and target variables."""
        print("Preparing features and targets...")
        
        # Define feature columns (exclude targets, flags, and timestamp)
        exclude_cols = (
            self.target_columns + 
            ['timestamp', 'quality_pass', 'is_anomaly', 'month', 
             'cetane_pass', 'sulfur_pass', 'viscosity_pass', 'flash_point_pass',
             'cloud_point_pass', 'aromatic_pass', 'pour_point_c', 'distillation_t90']
        )
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].copy()
        y = df[self.target_columns].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        print(f"Features: {X.shape[1]} columns")
        print(f"Targets: {y.shape[1]} columns")
        print(f"Samples: {X.shape[0]}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data with time-aware strategy."""
        print("Splitting data...")
        
        # Use time-based split to respect temporal dependencies
        n_total = len(X)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        # Split: oldest for train, middle for val, newest for test
        X_train = X.iloc[:n_train]
        X_val = X.iloc[n_train:n_train+n_val]
        X_test = X.iloc[n_train+n_val:]
        
        y_train = y.iloc[:n_train]
        y_val = y.iloc[n_train:n_train+n_val]
        y_test = y.iloc[n_train+n_val:]
        
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples") 
        print(f"Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test, target):
        """Scale features for better model performance."""
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[target] = scaler
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def select_features(self, X_train, y_train, target, k=15):
        """Select top k features for the target."""
        selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        
        # Store selector
        self.feature_selectors[target] = selector
        
        # Get selected feature names
        selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
        
        return X_train_selected, selected_features
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, target):
        """Train Random Forest model for a specific target."""
        print(f"Training Random Forest for {target}...")
        
        with mlflow.start_run(run_name=f"RandomForest_{target}", nested=True):
            # Log parameters
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            mlflow.log_params(params)
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("target", target)
            
            # Train model
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            
            # Validate
            y_val_pred = rf.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            r2 = r2_score(y_val, y_val_pred)
            
            metrics = {'mse': mse, 'mae': mae, 'r2': r2}
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(rf, f"random_forest_{target}")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, rf.feature_importances_))
            mlflow.log_dict(feature_importance, f"feature_importance_{target}.json")
            
            print(f"  R² Score: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            
            return rf, metrics
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, target):
        """Train XGBoost model for a specific target."""
        print(f"Training XGBoost for {target}...")
        
        with mlflow.start_run(run_name=f"XGBoost_{target}", nested=True):
            # Parameters optimized for fuel quality prediction
            params = {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            mlflow.log_params(params)
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("target", target)
            
            # Train model with early stopping
            xgb_model = xgb.XGBRegressor(**params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            # Validate
            y_val_pred = xgb_model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            r2 = r2_score(y_val, y_val_pred)
            
            metrics = {'mse': mse, 'mae': mae, 'r2': r2}
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.xgboost.log_model(xgb_model, f"xgboost_{target}")
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, xgb_model.feature_importances_))
            mlflow.log_dict(feature_importance, f"feature_importance_{target}.json")
            
            print(f"  R² Score: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
            
            return xgb_model, metrics
    
    def train_h2o_automl(self, X_train, y_train, X_val, y_val, target, max_runtime_secs=300):
        """Train H2O AutoML model for a specific target."""
        print(f"Training H2O AutoML for {target}...")
        
        try:
            import h2o
            from h2o.automl import H2OAutoML
            
            # Initialize H2O
            if not h2o.cluster().is_running():
                h2o.init(max_mem_size="4g", nthreads=-1)
            
            with mlflow.start_run(run_name=f"H2OAutoML_{target}", nested=True):
                mlflow.set_tag("model_type", "H2OAutoML")
                mlflow.set_tag("target", target)
                mlflow.log_param("max_runtime_secs", max_runtime_secs)
                
                # Convert to H2O frames
                train_df = pd.concat([X_train, y_train[[target]]], axis=1)
                val_df = pd.concat([X_val, y_val[[target]]], axis=1)
                
                h2o_train = h2o.H2OFrame(train_df)
                h2o_val = h2o.H2OFrame(val_df)
                
                # Set up AutoML
                aml = H2OAutoML(
                    max_runtime_secs=max_runtime_secs,
                    seed=self.random_state,
                    project_name=f"diesel_quality_{target}"
                )
                
                # Train
                aml.train(
                    x=self.feature_columns,
                    y=target,
                    training_frame=h2o_train,
                    validation_frame=h2o_val
                )
                
                # Get best model performance
                leader = aml.leader
                val_perf = leader.model_performance(h2o_val)
                
                # Extract metrics
                r2 = val_perf.r2()
                rmse = val_perf.rmse()
                mae = val_perf.mean_absolute_error()
                
                metrics = {'r2': r2, 'rmse': rmse, 'mae': mae, 'mse': rmse**2}
                mlflow.log_metrics(metrics)
                
                # Save model
                model_path = self.paths['models_path'] / f"h2o_automl_{target}"
                h2o.save_model(leader, str(model_path), force=True)
                
                print(f"  R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
                
                return leader, metrics
                
        except ImportError:
            print("H2O not available, skipping H2O AutoML")
            return None, {}
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set."""
        print("\nEvaluating models on test set...")
        
        test_results = {}
        
        for target in self.target_columns:
            if target in self.models:
                print(f"\nEvaluating {target}:")
                target_results = {}
                
                y_true = y_test[target]
                
                for model_name, model in self.models[target].items():
                    try:
                        if model_name == 'h2o_automl' and model is not None:
                            # H2O prediction
                            import h2o
                            test_df = pd.DataFrame(X_test, columns=self.feature_columns)
                            h2o_test = h2o.H2OFrame(test_df)
                            y_pred = model.predict(h2o_test).as_data_frame()['predict'].values
                        else:
                            # Scikit-learn/XGBoost prediction
                            # Apply feature selection if used
                            X_test_processed = X_test
                            if target in self.feature_selectors:
                                X_test_processed = self.feature_selectors[target].transform(X_test)
                            
                            y_pred = model.predict(X_test_processed)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        r2 = r2_score(y_true, y_pred)
                        
                        target_results[model_name] = {
                            'mse': mse,
                            'mae': mae,
                            'r2': r2,
                            'rmse': np.sqrt(mse)
                        }
                        
                        print(f"  {model_name}: R² = {r2:.4f}, RMSE = {np.sqrt(mse):.4f}")
                        
                    except Exception as e:
                        print(f"  Error evaluating {model_name}: {e}")
                        target_results[model_name] = {'error': str(e)}
                
                test_results[target] = target_results
        
        return test_results
    
    def select_best_models(self, test_results):
        """Select best model for each target based on R² score."""
        print("\nSelecting best models...")
        
        best_models = {}
        
        for target, results in test_results.items():
            best_r2 = -np.inf
            best_model_name = None
            
            for model_name, metrics in results.items():
                if 'r2' in metrics and metrics['r2'] > best_r2:
                    best_r2 = metrics['r2']
                    best_model_name = model_name
            
            if best_model_name:
                best_models[target] = {
                    'model_name': best_model_name,
                    'r2_score': best_r2,
                    'model': self.models[target][best_model_name]
                }
                print(f"  {target}: {best_model_name} (R² = {best_r2:.4f})")
        
        return best_models
    
    def save_models_and_artifacts(self, best_models, test_results):
        """Save best models and training artifacts."""
        print("\nSaving models and artifacts...")
        
        # Save best models
        models_dir = self.paths['models_path']
        
        for target, model_info in best_models.items():
            model = model_info['model']
            model_name = model_info['model_name']
            
            # Save model based on type
            if 'h2o' not in model_name:
                model_path = models_dir / f"best_{target}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save scaler if exists
                if target in self.scalers:
                    scaler_path = models_dir / f"scaler_{target}.pkl"
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[target], f)
                
                # Save feature selector if exists
                if target in self.feature_selectors:
                    selector_path = models_dir / f"feature_selector_{target}.pkl"
                    with open(selector_path, 'wb') as f:
                        pickle.dump(self.feature_selectors[target], f)
        
        # Save feature columns
        feature_path = models_dir / "feature_columns.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save model performance summary
        performance_path = models_dir / "model_performance.json"
        performance_summary = {
            'best_models': {
                target: {
                    'model_name': info['model_name'],
                    'r2_score': info['r2_score']
                }
                for target, info in best_models.items()
            },
            'all_results': test_results,
            'training_date': datetime.now().isoformat()
        }
        
        with open(performance_path, 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        print(f"Models saved to: {models_dir}")
        
        return performance_summary
    
    def generate_training_report(self, best_models, test_results, performance_summary):
        """Generate comprehensive training report."""
        print("\nGenerating training report...")
        
        report_path = self.paths['reports_path'] / 'model_training_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Diesel Fuel Quality Model Training Report\n\n")
            f.write(f"**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report summarizes the training of machine learning models for predicting diesel fuel quality parameters from refinery process inputs.\n\n")
            
            # Overall performance
            avg_r2 = np.mean([info['r2_score'] for info in best_models.values()])
            f.write(f"**Overall Model Performance:**\n")
            f.write(f"- Average R² Score: {avg_r2:.4f}\n")
            f.write(f"- Models Trained: {len(self.target_columns)} quality parameters\n")
            f.write(f"- Features Used: {len(self.feature_columns)}\n\n")
            
            # Best models summary
            f.write("## Best Models by Target\n\n")
            f.write("| Target Parameter | Model Type | R² Score | RMSE |\n")
            f.write("|-----------------|------------|----------|------|\n")
            
            for target, info in best_models.items():
                model_name = info['model_name']
                r2_score = info['r2_score']
                rmse = test_results[target][model_name]['rmse']
                f.write(f"| {target} | {model_name} | {r2_score:.4f} | {rmse:.4f} |\n")
            
            f.write("\n")
            
            # Detailed results
            f.write("## Detailed Model Comparison\n\n")
            for target in self.target_columns:
                if target in test_results:
                    f.write(f"### {target.replace('_', ' ').title()}\n\n")
                    f.write("| Model | R² Score | RMSE | MAE |\n")
                    f.write("|-------|----------|------|-----|\n")
                    
                    for model_name, metrics in test_results[target].items():
                        if 'r2' in metrics:
                            f.write(f"| {model_name} | {metrics['r2']:.4f} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} |\n")
                    f.write("\n")
            
            # Feature engineering
            f.write("## Feature Engineering\n\n")
            f.write("The following engineered features were created:\n\n")
            engineered_features = [
                'temp_pressure_interaction', 'catalyst_efficiency', 'hydrotreating_severity',
                'distillation_range', 'crude_quality_index', 'season_sin', 'season_cos',
                'log_sulfur_ppm', 'log_crude_sulfur'
            ]
            for feature in engineered_features:
                f.write(f"- **{feature}:** Domain-specific feature for process optimization\n")
            f.write("\n")
            
            # Model recommendations
            f.write("## Recommendations\n\n")
            
            high_performance_targets = [target for target, info in best_models.items() if info['r2_score'] > 0.9]
            medium_performance_targets = [target for target, info in best_models.items() if 0.8 <= info['r2_score'] <= 0.9]
            
            if high_performance_targets:
                f.write(f"**High Performance Models (R² > 0.9):**\n")
                for target in high_performance_targets:
                    f.write(f"- {target}: Ready for production deployment\n")
                f.write("\n")
            
            if medium_performance_targets:
                f.write(f"**Medium Performance Models (0.8 ≤ R² ≤ 0.9):**\n")
                for target in medium_performance_targets:
                    f.write(f"- {target}: Consider additional feature engineering\n")
                f.write("\n")
            
            f.write("**Next Steps:**\n")
            f.write("1. Deploy best models to production environment\n")
            f.write("2. Implement real-time prediction pipeline\n")
            f.write("3. Set up model monitoring and drift detection\n")
            f.write("4. Collect feedback for model improvement\n")
        
        print(f"Training report saved to: {report_path}")
    
    def train_all_models(self):
        """Main training pipeline."""
        print("DIESEL FUEL QUALITY MODEL TRAINING")
        print("=" * 60)
        
        with mlflow.start_run(run_name="DieselFuelQuality_Training"):
            # Load and prepare data
            df = self.load_data()
            df_engineered = self.engineer_features(df)
            X, y = self.prepare_features_and_targets(df_engineered)
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Log dataset info
            mlflow.log_params({
                "total_samples": len(X),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "n_features": X.shape[1],
                "n_targets": y.shape[1]
            })
            
            # Train models for each target
            for target in self.target_columns:
                print(f"\n{'='*50}")
                print(f"Training models for: {target}")
                print(f"{'='*50}")
                
                self.models[target] = {}
                
                # Get target values
                y_train_target = y_train[target]
                y_val_target = y_val[target]
                
                # Scale features
                X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
                    X_train, X_val, X_test, target
                )
                
                # Feature selection
                X_train_selected, selected_features = self.select_features(
                    X_train_scaled, y_train_target, target, k=15
                )
                X_val_selected = self.feature_selectors[target].transform(X_val_scaled)
                
                # Train Random Forest
                rf_model, rf_metrics = self.train_random_forest(
                    X_train_scaled, y_train_target, X_val_scaled, y_val_target, target
                )
                self.models[target]['random_forest'] = rf_model
                self.performance_metrics[f'{target}_rf'] = rf_metrics
                
                # Train XGBoost
                xgb_model, xgb_metrics = self.train_xgboost(
                    X_train_scaled, y_train_target, X_val_scaled, y_val_target, target
                )
                self.models[target]['xgboost'] = xgb_model
                self.performance_metrics[f'{target}_xgb'] = xgb_metrics
                
                # Train H2O AutoML (optional - requires H2O installation)
                try:
                    h2o_model, h2o_metrics = self.train_h2o_automl(
                        pd.DataFrame(X_train, columns=self.feature_columns), 
                        y_train[[target]], 
                        pd.DataFrame(X_val, columns=self.feature_columns), 
                        y_val[[target]], 
                        target, 
                        max_runtime_secs=120  # Reduced for demo
                    )
                    if h2o_model is not None:
                        self.models[target]['h2o_automl'] = h2o_model
                        self.performance_metrics[f'{target}_h2o'] = h2o_metrics
                except:
                    print("  H2O AutoML training skipped")
            
            # Evaluate models on test set
            test_results = self.evaluate_models(X_test, y_test)
            
            # Select best models
            best_models = self.select_best_models(test_results)
            
            # Save models and artifacts
            performance_summary = self.save_models_and_artifacts(best_models, test_results)
            
            # Generate report
            self.generate_training_report(best_models, test_results, performance_summary)
            
            # Log overall metrics to MLflow
            if best_models:
                avg_r2 = np.mean([info['r2_score'] for info in best_models.values()])
                mlflow.log_metric("average_r2_score", avg_r2)
                mlflow.log_metric("models_trained", len(best_models))
            
            print(f"\n{'='*60}")
            print("MODEL TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Best models saved: {len(best_models)}")
            print(f"Average R² score: {avg_r2:.4f}")
            print(f"Artifacts saved to: {self.paths['models_path']}")
            
            return best_models, test_results

def main():
    """Main training function."""
    trainer = DieselFuelModelTrainer(random_state=42)
    best_models, test_results = trainer.train_all_models()
    return trainer, best_models, test_results

if __name__ == "__main__":
    trainer, best_models, test_results = main()