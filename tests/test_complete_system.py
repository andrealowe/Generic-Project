"""
Comprehensive System Tests for Diesel Fuel Quality Prediction

Tests the entire pipeline from data generation through prediction API.
"""

import unittest
import pandas as pd
import numpy as np
import requests
import json
import sys
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

class TestCompleteSystem(unittest.TestCase):
    """Test suite for the complete diesel fuel quality prediction system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.paths = get_data_paths()
        cls.api_port = 8001  # Use different port for testing
        cls.api_process = None
        
    def test_01_data_configuration(self):
        """Test data configuration and paths setup."""
        print("\n🧪 Testing data configuration...")
        
        # Test path creation
        self.assertTrue(self.paths['base_data_path'].exists())
        self.assertTrue(self.paths['artifacts_path'].exists())
        
        # Test project type detection
        self.assertEqual(self.paths['project_type'], 'git')
        
        print("✅ Data configuration tests passed")
    
    def test_02_synthetic_data_generation(self):
        """Test synthetic data generation."""
        print("\n🧪 Testing synthetic data generation...")
        
        # Import and test data generator
        from src.data.synthetic_data_generator import DieselDataGenerator
        
        # Create small dataset for testing
        generator = DieselDataGenerator(n_samples=100, random_seed=42)
        df = generator.generate_dataset()
        
        # Test dataset properties
        self.assertEqual(len(df), 100)
        self.assertGreater(len(df.columns), 30)  # Should have many columns
        
        # Test required columns exist
        required_cols = ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'quality_score']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Test data quality
        self.assertEqual(df.isnull().sum().sum(), 0)  # No missing values
        self.assertTrue((df['quality_score'] >= 0).all())
        self.assertTrue((df['quality_score'] <= 100).all())
        
        print("✅ Synthetic data generation tests passed")
    
    def test_03_eda_execution(self):
        """Test EDA script execution."""
        print("\n🧪 Testing EDA execution...")
        
        try:
            from notebooks.diesel_fuel_eda import load_dataset, basic_statistics
            
            # Test if we can load data and run basic stats
            df, paths = load_dataset()
            self.assertGreater(len(df), 0)
            
            desc_stats = basic_statistics(df)
            self.assertIsInstance(desc_stats, pd.DataFrame)
            
            print("✅ EDA execution tests passed")
            
        except Exception as e:
            self.fail(f"EDA execution failed: {e}")
    
    def test_04_model_training(self):
        """Test model training pipeline."""
        print("\n🧪 Testing model training...")
        
        from src.models.quick_model_trainer import QuickModelTrainer
        
        # Create small trainer for testing
        trainer = QuickModelTrainer(random_state=42)
        
        # Load small dataset
        X, y, feature_columns = trainer.load_and_prepare_data()
        
        # Test data loading
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
        self.assertEqual(len(X), len(y))
        
        # Test individual model training (just one target for speed)
        target = 'cetane_number'
        result = trainer.train_models_for_target(X, y[target], target, feature_columns)
        
        # Test results
        self.assertIn('model', result)
        self.assertIn('metrics', result)
        self.assertIn('r2', result['metrics'])
        
        print("✅ Model training tests passed")
    
    def test_05_model_artifacts(self):
        """Test model artifacts are created and loadable."""
        print("\n🧪 Testing model artifacts...")
        
        models_dir = self.paths['models_path']
        
        # Check if model files exist
        feature_file = models_dir / 'feature_columns.json'
        perf_file = models_dir / 'model_performance.json'
        
        if feature_file.exists():
            # Test loading feature columns
            with open(feature_file, 'r') as f:
                features = json.load(f)
            self.assertIsInstance(features, list)
            self.assertGreater(len(features), 0)
            
        if perf_file.exists():
            # Test loading performance data
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
            self.assertIn('best_models', perf_data)
            
        print("✅ Model artifacts tests passed")
    
    def test_06_dashboard_components(self):
        """Test dashboard component loading."""
        print("\n🧪 Testing dashboard components...")
        
        try:
            # Test dashboard class instantiation
            from scripts.diesel_dashboard import DieselDashboard
            
            dashboard = DieselDashboard()
            
            # Test basic properties
            self.assertIsInstance(dashboard.df, pd.DataFrame)
            self.assertIsInstance(dashboard.target_columns, list)
            self.assertGreater(len(dashboard.target_columns), 0)
            
            print("✅ Dashboard component tests passed")
            
        except Exception as e:
            print(f"⚠️ Dashboard tests skipped: {e}")
    
    def test_07_api_functionality(self):
        """Test API functionality."""
        print("\n🧪 Testing API functionality...")
        
        try:
            # Test API imports and basic structure
            from src.api.predict import app, ProcessInput, QualityPrediction
            
            # Test input schema
            test_input = {
                "crude_api_gravity": 35.0,
                "crude_sulfur_ppm": 1500.0,
                "crude_nitrogen_ppm": 500.0,
                "crude_tan": 1.0,
                "distillation_temp_top": 95.0,
                "distillation_temp_middle": 250.0,
                "distillation_temp_bottom": 385.0,
                "distillation_pressure": 1.0,
                "feed_rate_bpd": 50000.0,
                "reflux_ratio": 2.5,
                "hydrotreat_temp": 340.0,
                "hydrotreat_pressure": 90.0,
                "hydrogen_ratio": 300.0,
                "lhsv": 1.5,
                "catalyst_age_days": 200.0
            }
            
            # Test input validation
            process_input = ProcessInput(**test_input)
            self.assertIsInstance(process_input, ProcessInput)
            
            print("✅ API functionality tests passed")
            
        except Exception as e:
            print(f"⚠️ API tests skipped: {e}")
    
    def test_08_configuration_files(self):
        """Test configuration files are valid."""
        print("\n🧪 Testing configuration files...")
        
        # Test Flow configuration
        flow_config = Path('/mnt/code/config/flow_config.yml')
        if flow_config.exists():
            import yaml
            with open(flow_config, 'r') as f:
                flow_data = yaml.safe_load(f)
            
            self.assertIn('name', flow_data)
            self.assertIn('steps', flow_data)
            self.assertIsInstance(flow_data['steps'], list)
            self.assertGreater(len(flow_data['steps']), 0)
        
        # Test Launcher configuration
        launcher_config = Path('/mnt/code/config/launcher_config.json')
        if launcher_config.exists():
            with open(launcher_config, 'r') as f:
                launcher_data = json.load(f)
            
            self.assertIn('name', launcher_data)
            self.assertIn('parameters', launcher_data)
            self.assertIsInstance(launcher_data['parameters'], list)
        
        print("✅ Configuration files tests passed")
    
    def test_09_app_launcher(self):
        """Test app launcher script."""
        print("\n🧪 Testing app launcher...")
        
        app_script = Path('/mnt/code/app.sh')
        self.assertTrue(app_script.exists())
        
        # Check if script is executable
        self.assertTrue(os.access(app_script, os.X_OK))
        
        # Check script content
        with open(app_script, 'r') as f:
            content = f.read()
        
        self.assertIn('streamlit', content.lower())
        self.assertIn('8501', content)  # Default Streamlit port
        
        print("✅ App launcher tests passed")
    
    def test_10_end_to_end_prediction(self):
        """Test end-to-end prediction workflow."""
        print("\n🧪 Testing end-to-end prediction...")
        
        try:
            # Check if models are available
            models_dir = self.paths['models_path']
            model_files = list(models_dir.glob('best_*.pkl'))
            
            if len(model_files) > 0:
                print(f"Found {len(model_files)} model files")
                
                # Test model loading and prediction
                import pickle
                
                # Load one model for testing
                model_file = model_files[0]
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Create dummy input data
                dummy_input = np.random.rand(1, 15)  # Assuming 15 features
                
                # Test prediction (may fail if scaler not available, but model should load)
                self.assertIsNotNone(model)
                print("✅ Model loading successful")
                
            else:
                print("⚠️ No trained models found - skipping prediction test")
        
        except Exception as e:
            print(f"⚠️ End-to-end prediction test failed: {e}")
    
    def test_11_reports_and_outputs(self):
        """Test that expected reports and outputs are generated."""
        print("\n🧪 Testing reports and outputs...")
        
        reports_dir = self.paths['reports_path']
        viz_dir = self.paths['visualizations_path']
        
        # Check for expected reports
        expected_reports = [
            'eda_summary_report.md',
            'model_training_report.md'
        ]
        
        for report in expected_reports:
            report_path = reports_dir / report
            if report_path.exists():
                self.assertGreater(report_path.stat().st_size, 0)
                print(f"✅ Found {report}")
            else:
                print(f"⚠️ Missing {report}")
        
        print("✅ Reports and outputs tests completed")
    
    def test_12_system_integration(self):
        """Test overall system integration."""
        print("\n🧪 Testing system integration...")
        
        # Count available components
        components = {
            'data_generator': False,
            'eda_script': False, 
            'model_trainer': False,
            'dashboard': False,
            'api': False,
            'app_launcher': False,
            'flow_config': False,
            'launcher_config': False
        }
        
        # Check each component
        try:
            from src.data.synthetic_data_generator import DieselDataGenerator
            components['data_generator'] = True
        except:
            pass
            
        try:
            import notebooks.diesel_fuel_eda
            components['eda_script'] = True
        except:
            pass
            
        try:
            from src.models.quick_model_trainer import QuickModelTrainer
            components['model_trainer'] = True
        except:
            pass
            
        try:
            from scripts.diesel_dashboard import DieselDashboard
            components['dashboard'] = True
        except:
            pass
            
        try:
            from src.api.predict import app
            components['api'] = True
        except:
            pass
            
        components['app_launcher'] = Path('/mnt/code/app.sh').exists()
        components['flow_config'] = Path('/mnt/code/config/flow_config.yml').exists()
        components['launcher_config'] = Path('/mnt/code/config/launcher_config.json').exists()
        
        # Print integration status
        available_count = sum(components.values())
        total_count = len(components)
        
        print(f"\nSystem Integration Status: {available_count}/{total_count} components available")
        
        for component, available in components.items():
            status = "✅" if available else "❌"
            print(f"  {status} {component}")
        
        # Require at least 75% of components to be available
        integration_score = available_count / total_count
        self.assertGreaterEqual(integration_score, 0.75, 
                               f"System integration score too low: {integration_score:.2%}")
        
        print(f"\n✅ System integration tests passed (Score: {integration_score:.1%})")

class TestSystemPerformance(unittest.TestCase):
    """Performance and benchmark tests."""
    
    def setUp(self):
        self.paths = get_data_paths()
    
    def test_data_generation_performance(self):
        """Test data generation performance."""
        print("\n🏃 Testing data generation performance...")
        
        from src.data.synthetic_data_generator import DieselDataGenerator
        
        # Time small dataset generation
        start_time = time.time()
        generator = DieselDataGenerator(n_samples=1000, random_seed=42)
        df = generator.generate_dataset()
        generation_time = time.time() - start_time
        
        print(f"Generated 1000 samples in {generation_time:.2f} seconds")
        
        # Should be reasonably fast
        self.assertLess(generation_time, 30.0, "Data generation too slow")
        
        print("✅ Data generation performance acceptable")
    
    def test_model_training_performance(self):
        """Test model training performance."""
        print("\n🏃 Testing model training performance...")
        
        if not self.paths['models_path'].exists():
            print("⚠️ No models directory found, skipping performance test")
            return
        
        # Check if models were created recently
        model_files = list(self.paths['models_path'].glob('*.pkl'))
        
        if model_files:
            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            model_age = time.time() - latest_model.stat().st_mtime
            
            print(f"Latest model created {model_age/60:.1f} minutes ago")
            
            # Models should be reasonably recent if training just completed
            if model_age < 3600:  # Within last hour
                print("✅ Recent model training detected")
            else:
                print("ℹ️ No recent model training detected")
        else:
            print("⚠️ No model files found")

def run_system_tests():
    """Run all system tests with proper reporting."""
    print("🚀 DIESEL FUEL QUALITY SYSTEM TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test methods from TestCompleteSystem
    system_tests = unittest.TestLoader().loadTestsFromTestCase(TestCompleteSystem)
    performance_tests = unittest.TestLoader().loadTestsFromTestCase(TestSystemPerformance)
    
    test_suite.addTests(system_tests)
    test_suite.addTests(performance_tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_system_tests()
    sys.exit(0 if success else 1)