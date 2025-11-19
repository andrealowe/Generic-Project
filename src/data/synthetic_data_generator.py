"""
Synthetic Diesel Fuel Quality Data Generator

Generates realistic synthetic data for diesel fuel quality prediction based on:
- ASTM D975 specifications
- Refinery process relationships
- Industry standards and correlations

Dataset includes:
- Process input features (atmospheric distillation, hydrotreating, crude properties)
- Target quality parameters (cetane number, sulfur content, viscosity, etc.)
- Realistic correlations and noise
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, '/mnt/code')
from scripts.data_config import get_data_paths

logger = logging.getLogger(__name__)

class DieselDataGenerator:
    """Generate synthetic diesel fuel quality dataset with realistic correlations."""
    
    def __init__(self, n_samples=8000, random_seed=42):
        """
        Initialize the data generator.
        
        Args:
            n_samples (int): Number of samples to generate
            random_seed (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Industry-standard ranges based on ASTM D975 and refinery operations
        self.parameter_ranges = {
            # Crude oil properties
            'crude_api_gravity': (25, 45),  # °API
            'crude_sulfur_ppm': (100, 8000),  # ppm
            'crude_nitrogen_ppm': (50, 2000),  # ppm
            'crude_tan': (0.1, 3.0),  # Total Acid Number
            
            # Atmospheric distillation
            'distillation_temp_top': (80, 120),  # °C
            'distillation_temp_middle': (220, 280),  # °C
            'distillation_temp_bottom': (350, 420),  # °C
            'distillation_pressure': (0.9, 1.1),  # bar (slightly above atmospheric)
            'feed_rate_bpd': (10000, 100000),  # barrels per day
            'reflux_ratio': (1.5, 4.0),  # dimensionless
            
            # Hydrotreating process
            'hydrotreat_temp': (300, 380),  # °C
            'hydrotreat_pressure': (40, 140),  # bar
            'hydrogen_ratio': (200, 500),  # SCF/BBL
            'lhsv': (0.8, 3.0),  # hr⁻¹ (Liquid Hourly Space Velocity)
            'catalyst_age_days': (0, 1095),  # days (3 years max)
            
            # Target quality parameters (ASTM D975)
            'cetane_number': (40, 60),  # dimensionless
            'sulfur_ppm': (1, 500),  # ppm (ULSD: ≤15)
            'viscosity_cst': (1.9, 4.1),  # mm²/s at 40°C
            'flash_point_c': (52, 85),  # °C
            'density_gcc': (0.82, 0.86),  # g/cm³
            'cloud_point_c': (-20, 5),  # °C
            'pour_point_c': (-35, -5),  # °C
            'distillation_t90': (280, 370),  # °C (T90)
            'aromatic_content': (10, 35),  # % volume
        }
    
    def generate_base_features(self):
        """Generate base process input features."""
        data = {}
        
        # Generate crude oil properties
        data['crude_api_gravity'] = np.random.uniform(*self.parameter_ranges['crude_api_gravity'], self.n_samples)
        data['crude_sulfur_ppm'] = np.random.uniform(*self.parameter_ranges['crude_sulfur_ppm'], self.n_samples)
        data['crude_nitrogen_ppm'] = np.random.uniform(*self.parameter_ranges['crude_nitrogen_ppm'], self.n_samples)
        data['crude_tan'] = np.random.uniform(*self.parameter_ranges['crude_tan'], self.n_samples)
        
        # Generate distillation parameters
        data['distillation_temp_top'] = np.random.uniform(*self.parameter_ranges['distillation_temp_top'], self.n_samples)
        data['distillation_temp_middle'] = np.random.uniform(*self.parameter_ranges['distillation_temp_middle'], self.n_samples)
        data['distillation_temp_bottom'] = np.random.uniform(*self.parameter_ranges['distillation_temp_bottom'], self.n_samples)
        data['distillation_pressure'] = np.random.uniform(*self.parameter_ranges['distillation_pressure'], self.n_samples)
        data['feed_rate_bpd'] = np.random.uniform(*self.parameter_ranges['feed_rate_bpd'], self.n_samples)
        data['reflux_ratio'] = np.random.uniform(*self.parameter_ranges['reflux_ratio'], self.n_samples)
        
        # Generate hydrotreating parameters
        data['hydrotreat_temp'] = np.random.uniform(*self.parameter_ranges['hydrotreat_temp'], self.n_samples)
        data['hydrotreat_pressure'] = np.random.uniform(*self.parameter_ranges['hydrotreat_pressure'], self.n_samples)
        data['hydrogen_ratio'] = np.random.uniform(*self.parameter_ranges['hydrogen_ratio'], self.n_samples)
        data['lhsv'] = np.random.uniform(*self.parameter_ranges['lhsv'], self.n_samples)
        data['catalyst_age_days'] = np.random.uniform(*self.parameter_ranges['catalyst_age_days'], self.n_samples)
        
        # Add timestamps (simulate 3 years of hourly data)
        start_date = datetime(2021, 1, 1)
        timestamps = [start_date + timedelta(hours=i * (3 * 365 * 24 / self.n_samples)) 
                     for i in range(self.n_samples)]
        data['timestamp'] = timestamps
        
        return pd.DataFrame(data)
    
    def calculate_quality_parameters(self, df):
        """Calculate target quality parameters based on process inputs with realistic correlations."""
        
        # Normalize inputs for calculations
        api_norm = (df['crude_api_gravity'] - 25) / 20
        sulfur_norm = (df['crude_sulfur_ppm'] - 100) / 7900
        ht_temp_norm = (df['hydrotreat_temp'] - 300) / 80
        ht_pressure_norm = (df['hydrotreat_pressure'] - 40) / 100
        
        # 1. Cetane Number (higher API gravity = more paraffinic = higher cetane)
        cetane_base = 45 + 8 * api_norm - 3 * sulfur_norm
        cetane_distillation_effect = 2 * (df['distillation_temp_middle'] - 250) / 30
        df['cetane_number'] = np.clip(
            cetane_base + cetane_distillation_effect + np.random.normal(0, 2, len(df)),
            *self.parameter_ranges['cetane_number']
        )
        
        # 2. Sulfur Content (decreases with hydrotreating severity)
        # Arrhenius-type relationship
        sulfur_reduction_factor = np.exp(-0.015 * (df['hydrotreat_temp'] - 320) - 0.005 * ht_pressure_norm * 100)
        catalyst_efficiency = np.exp(-df['catalyst_age_days'] / 800)  # Catalyst deactivation
        
        df['sulfur_ppm'] = np.clip(
            df['crude_sulfur_ppm'] * sulfur_reduction_factor * catalyst_efficiency + 
            np.random.normal(0, 3, len(df)),
            *self.parameter_ranges['sulfur_ppm']
        )
        
        # 3. Viscosity (related to molecular weight distribution)
        viscosity_base = 2.8 + 0.5 * (df['distillation_temp_middle'] - 250) / 30
        viscosity_api_effect = -0.3 * api_norm
        df['viscosity_cst'] = np.clip(
            viscosity_base + viscosity_api_effect + np.random.normal(0, 0.15, len(df)),
            *self.parameter_ranges['viscosity_cst']
        )
        
        # 4. Flash Point (safety parameter)
        flash_base = 60 + 0.8 * (df['distillation_temp_bottom'] - 380) / 40
        df['flash_point_c'] = np.clip(
            flash_base + np.random.normal(0, 3, len(df)),
            *self.parameter_ranges['flash_point_c']
        )
        
        # 5. Density (related to API gravity)
        df['density_gcc'] = np.clip(
            141.5 / (df['crude_api_gravity'] + 131.5) + np.random.normal(0, 0.005, len(df)),
            *self.parameter_ranges['density_gcc']
        )
        
        # 6. Cloud Point (wax crystallization temperature)
        cloud_base = -10 + 15 * (1 - api_norm) 
        df['cloud_point_c'] = np.clip(
            cloud_base + np.random.normal(0, 3, len(df)),
            *self.parameter_ranges['cloud_point_c']
        )
        
        # 7. Pour Point (always lower than cloud point)
        df['pour_point_c'] = np.clip(
            df['cloud_point_c'] - np.random.uniform(10, 25, len(df)),
            *self.parameter_ranges['pour_point_c']
        )
        
        # 8. Distillation T90
        t90_base = 320 + 0.6 * (df['distillation_temp_bottom'] - 380) / 40
        df['distillation_t90'] = np.clip(
            t90_base + np.random.normal(0, 8, len(df)),
            *self.parameter_ranges['distillation_t90']
        )
        
        # 9. Aromatic Content (affects emissions)
        aromatic_base = 25 - 10 * api_norm + 5 * sulfur_norm
        df['aromatic_content'] = np.clip(
            aromatic_base + np.random.normal(0, 3, len(df)),
            *self.parameter_ranges['aromatic_content']
        )
        
        return df
    
    def add_quality_flags(self, df):
        """Add quality pass/fail flags based on ASTM D975 specifications."""
        
        # ASTM D975 Grade No. 2-D specifications
        specs = {
            'cetane_number_min': 40,
            'sulfur_ppm_max': 15,  # ULSD specification
            'viscosity_min': 1.9,
            'viscosity_max': 4.1,
            'flash_point_min': 52,
            'cloud_point_max': -7,  # Winter specification (more stringent)
            'aromatic_content_max': 35
        }
        
        # Check individual specifications
        df['cetane_pass'] = df['cetane_number'] >= specs['cetane_number_min']
        df['sulfur_pass'] = df['sulfur_ppm'] <= specs['sulfur_ppm_max']
        df['viscosity_pass'] = ((df['viscosity_cst'] >= specs['viscosity_min']) & 
                               (df['viscosity_cst'] <= specs['viscosity_max']))
        df['flash_point_pass'] = df['flash_point_c'] >= specs['flash_point_min']
        df['cloud_point_pass'] = df['cloud_point_c'] <= specs['cloud_point_max']
        df['aromatic_pass'] = df['aromatic_content'] <= specs['aromatic_content_max']
        
        # Overall quality flag (all specifications must pass)
        df['quality_pass'] = (df['cetane_pass'] & df['sulfur_pass'] & 
                             df['viscosity_pass'] & df['flash_point_pass'] & 
                             df['cloud_point_pass'] & df['aromatic_pass'])
        
        # Quality score (0-100)
        quality_scores = []
        for _, row in df.iterrows():
            score = 0
            score += 20 if row['cetane_pass'] else max(0, 20 * row['cetane_number'] / 40)
            score += 20 if row['sulfur_pass'] else max(0, 20 * (100 - row['sulfur_ppm']) / 100)
            score += 15 if row['viscosity_pass'] else 10
            score += 15 if row['flash_point_pass'] else max(0, 15 * row['flash_point_c'] / 52)
            score += 15 if row['cloud_point_pass'] else 10
            score += 15 if row['aromatic_pass'] else max(0, 15 * (50 - row['aromatic_content']) / 50)
            quality_scores.append(min(100, score))
        
        df['quality_score'] = quality_scores
        
        return df
    
    def add_seasonal_variations(self, df):
        """Add seasonal variations to simulate real-world operations."""
        
        # Extract month for seasonal effects
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Winter diesel specifications (Nov-Mar) - lower cloud point requirements
        winter_months = [11, 12, 1, 2, 3]
        is_winter = df['month'].isin(winter_months)
        
        # Adjust cloud point for seasonal blending
        df.loc[is_winter, 'cloud_point_c'] = df.loc[is_winter, 'cloud_point_c'] - 5
        df.loc[is_winter, 'pour_point_c'] = df.loc[is_winter, 'pour_point_c'] - 3
        
        # Summer blend adjustments (Apr-Oct) - vapor pressure considerations
        summer_months = [4, 5, 6, 7, 8, 9, 10]
        is_summer = df['month'].isin(summer_months)
        
        # Slightly higher aromatic content allowed in summer
        df.loc[is_summer, 'aromatic_content'] = df.loc[is_summer, 'aromatic_content'] + 2
        
        return df
    
    def add_anomalies(self, df, anomaly_rate=0.05):
        """Add realistic anomalies to represent off-spec production."""
        
        n_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        df['is_anomaly'] = False
        df.loc[anomaly_indices, 'is_anomaly'] = True
        
        # Different types of anomalies
        for idx in anomaly_indices:
            anomaly_type = np.random.choice([
                'high_sulfur', 'low_cetane', 'viscosity_out', 
                'process_upset', 'catalyst_poisoning'
            ])
            
            if anomaly_type == 'high_sulfur':
                df.loc[idx, 'sulfur_ppm'] *= np.random.uniform(5, 15)
                df.loc[idx, 'quality_pass'] = False
                
            elif anomaly_type == 'low_cetane':
                df.loc[idx, 'cetane_number'] *= np.random.uniform(0.7, 0.9)
                df.loc[idx, 'quality_pass'] = False
                
            elif anomaly_type == 'viscosity_out':
                if np.random.random() > 0.5:
                    df.loc[idx, 'viscosity_cst'] *= np.random.uniform(1.3, 2.0)  # Too high
                else:
                    df.loc[idx, 'viscosity_cst'] *= np.random.uniform(0.4, 0.7)  # Too low
                df.loc[idx, 'quality_pass'] = False
                
            elif anomaly_type == 'process_upset':
                # Temperature excursion affects multiple parameters
                df.loc[idx, 'hydrotreat_temp'] *= np.random.uniform(1.1, 1.3)
                df.loc[idx, 'sulfur_ppm'] *= np.random.uniform(2, 8)
                df.loc[idx, 'cetane_number'] *= np.random.uniform(0.85, 0.95)
                df.loc[idx, 'quality_pass'] = False
                
            elif anomaly_type == 'catalyst_poisoning':
                # Gradual catalyst deactivation
                df.loc[idx, 'sulfur_ppm'] *= np.random.uniform(3, 10)
                df.loc[idx, 'aromatic_content'] *= np.random.uniform(1.2, 1.8)
                df.loc[idx, 'quality_pass'] = False
        
        # Recalculate quality score for anomalies
        df = self.add_quality_flags(df)
        
        return df
    
    def generate_dataset(self, save_path=None):
        """
        Generate complete synthetic diesel fuel quality dataset.
        
        Args:
            save_path (str): Optional path to save the dataset
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        
        logger.info(f"Generating {self.n_samples} synthetic diesel fuel quality samples...")
        
        # Generate base features
        df = self.generate_base_features()
        logger.info("Generated base process features")
        
        # Calculate quality parameters
        df = self.calculate_quality_parameters(df)
        logger.info("Calculated quality parameters")
        
        # Add quality flags and scores
        df = self.add_quality_flags(df)
        logger.info("Added quality specifications")
        
        # Add seasonal variations
        df = self.add_seasonal_variations(df)
        logger.info("Added seasonal variations")
        
        # Add anomalies
        df = self.add_anomalies(df)
        logger.info("Added realistic anomalies")
        
        # Final data cleaning
        df = df.round(4)  # Round to 4 decimal places for realism
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate statistics
        pass_rate = df['quality_pass'].mean() * 100
        avg_score = df['quality_score'].mean()
        
        logger.info(f"Dataset generation complete:")
        logger.info(f"  - Total samples: {len(df)}")
        logger.info(f"  - Quality pass rate: {pass_rate:.1f}%")
        logger.info(f"  - Average quality score: {avg_score:.1f}")
        logger.info(f"  - Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")
        
        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Dataset saved to: {save_path}")
            
            # Also save data dictionary
            self.save_data_dictionary(save_path.replace('.csv', '_dictionary.json'))
        
        return df
    
    def save_data_dictionary(self, filepath):
        """Save data dictionary with feature descriptions and ranges."""
        import json
        
        dictionary = {
            "dataset_info": {
                "name": "Synthetic Diesel Fuel Quality Dataset",
                "description": "Synthetic dataset for predicting diesel fuel quality from refinery process inputs",
                "n_samples": self.n_samples,
                "generation_date": datetime.now().isoformat(),
                "random_seed": self.random_seed,
                "standards_reference": "ASTM D975-24",
                "quality_pass_rate": "~85-90%",
                "anomaly_rate": "~5%"
            },
            "features": {
                # Process inputs
                "timestamp": {"description": "Sample timestamp", "type": "datetime", "unit": "ISO format"},
                "crude_api_gravity": {"description": "API gravity of crude oil feed", "type": "float", "unit": "°API", "range": self.parameter_ranges['crude_api_gravity']},
                "crude_sulfur_ppm": {"description": "Sulfur content in crude oil", "type": "float", "unit": "ppm", "range": self.parameter_ranges['crude_sulfur_ppm']},
                "crude_nitrogen_ppm": {"description": "Nitrogen content in crude oil", "type": "float", "unit": "ppm", "range": self.parameter_ranges['crude_nitrogen_ppm']},
                "crude_tan": {"description": "Total Acid Number of crude oil", "type": "float", "unit": "mg KOH/g", "range": self.parameter_ranges['crude_tan']},
                "distillation_temp_top": {"description": "Top tray temperature in distillation column", "type": "float", "unit": "°C", "range": self.parameter_ranges['distillation_temp_top']},
                "distillation_temp_middle": {"description": "Middle tray temperature in distillation column", "type": "float", "unit": "°C", "range": self.parameter_ranges['distillation_temp_middle']},
                "distillation_temp_bottom": {"description": "Bottom tray temperature in distillation column", "type": "float", "unit": "°C", "range": self.parameter_ranges['distillation_temp_bottom']},
                "distillation_pressure": {"description": "Operating pressure in distillation column", "type": "float", "unit": "bar", "range": self.parameter_ranges['distillation_pressure']},
                "feed_rate_bpd": {"description": "Crude oil feed rate", "type": "float", "unit": "barrels/day", "range": self.parameter_ranges['feed_rate_bpd']},
                "reflux_ratio": {"description": "Reflux ratio in distillation column", "type": "float", "unit": "dimensionless", "range": self.parameter_ranges['reflux_ratio']},
                "hydrotreat_temp": {"description": "Hydrotreating reactor temperature", "type": "float", "unit": "°C", "range": self.parameter_ranges['hydrotreat_temp']},
                "hydrotreat_pressure": {"description": "Hydrotreating reactor pressure", "type": "float", "unit": "bar", "range": self.parameter_ranges['hydrotreat_pressure']},
                "hydrogen_ratio": {"description": "Hydrogen to oil ratio", "type": "float", "unit": "SCF/BBL", "range": self.parameter_ranges['hydrogen_ratio']},
                "lhsv": {"description": "Liquid Hourly Space Velocity", "type": "float", "unit": "hr⁻¹", "range": self.parameter_ranges['lhsv']},
                "catalyst_age_days": {"description": "Age of hydrotreating catalyst", "type": "float", "unit": "days", "range": self.parameter_ranges['catalyst_age_days']},
                
                # Target quality parameters
                "cetane_number": {"description": "Cetane number (ignition quality)", "type": "float", "unit": "dimensionless", "range": self.parameter_ranges['cetane_number'], "target": True, "astm": "D613"},
                "sulfur_ppm": {"description": "Sulfur content in final product", "type": "float", "unit": "ppm", "range": self.parameter_ranges['sulfur_ppm'], "target": True, "astm": "D5453"},
                "viscosity_cst": {"description": "Kinematic viscosity at 40°C", "type": "float", "unit": "mm²/s", "range": self.parameter_ranges['viscosity_cst'], "target": True, "astm": "D445"},
                "flash_point_c": {"description": "Flash point (safety parameter)", "type": "float", "unit": "°C", "range": self.parameter_ranges['flash_point_c'], "target": True, "astm": "D93"},
                "density_gcc": {"description": "Density at 15°C", "type": "float", "unit": "g/cm³", "range": self.parameter_ranges['density_gcc'], "target": True, "astm": "D4052"},
                "cloud_point_c": {"description": "Cloud point (wax crystallization)", "type": "float", "unit": "°C", "range": self.parameter_ranges['cloud_point_c'], "target": True, "astm": "D2500"},
                "pour_point_c": {"description": "Pour point (flow temperature)", "type": "float", "unit": "°C", "range": self.parameter_ranges['pour_point_c'], "target": True, "astm": "D97"},
                "distillation_t90": {"description": "90% distillation temperature", "type": "float", "unit": "°C", "range": self.parameter_ranges['distillation_t90'], "target": True, "astm": "D86"},
                "aromatic_content": {"description": "Aromatic content", "type": "float", "unit": "% volume", "range": self.parameter_ranges['aromatic_content'], "target": True, "astm": "D1319"},
                
                # Quality flags
                "quality_pass": {"description": "Overall quality pass/fail", "type": "bool", "unit": "boolean", "target": True},
                "quality_score": {"description": "Overall quality score", "type": "float", "unit": "0-100", "target": True},
                "is_anomaly": {"description": "Anomaly flag", "type": "bool", "unit": "boolean", "target": False},
                "month": {"description": "Month for seasonal analysis", "type": "int", "unit": "1-12", "target": False}
            },
            "quality_specifications": {
                "ASTM_D975_Grade_2D": {
                    "cetane_number_min": 40,
                    "sulfur_ppm_max": 15,
                    "viscosity_range": [1.9, 4.1],
                    "flash_point_min": 52,
                    "cloud_point_max": -7,
                    "aromatic_content_max": 35
                }
            },
            "correlations": {
                "cetane_number": "Positively correlated with API gravity (paraffinic content), negatively with sulfur",
                "sulfur_ppm": "Decreases exponentially with hydrotreating temperature and pressure, affected by catalyst age",
                "viscosity_cst": "Related to molecular weight distribution, affected by distillation temperatures",
                "density_gcc": "Inversely related to API gravity (industry standard correlation)",
                "cloud_point_c": "Related to paraffinic content and molecular structure",
                "aromatic_content": "Affects emissions, inversely related to cetane number"
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(dictionary, f, indent=2)
        
        logger.info(f"Data dictionary saved to: {filepath}")

def main():
    """Main function to generate and save the dataset."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get data paths
    paths = get_data_paths()
    
    # Create data generator
    generator = DieselDataGenerator(n_samples=8000, random_seed=42)
    
    # Generate dataset
    dataset_path = paths['processed_data_path'] / 'diesel_quality_dataset.csv'
    df = generator.generate_dataset(save_path=str(dataset_path))
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DIESEL FUEL QUALITY DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Saved to: {dataset_path}")
    print(f"\nKey statistics:")
    print(f"  Quality pass rate: {df['quality_pass'].mean()*100:.1f}%")
    print(f"  Average quality score: {df['quality_score'].mean():.1f}")
    print(f"  Anomaly rate: {df['is_anomaly'].mean()*100:.1f}%")
    print(f"\nTarget quality ranges:")
    for param in ['cetane_number', 'sulfur_ppm', 'viscosity_cst', 'flash_point_c']:
        print(f"  {param}: {df[param].min():.2f} - {df[param].max():.2f}")
    
    return df

if __name__ == "__main__":
    df = main()