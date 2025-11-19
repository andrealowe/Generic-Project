"""
Data Configuration Utility for Domino Projects

This utility determines the correct data storage paths based on project type:
- Git-based projects: Data in $DOMINO_DATASETS_DIR/{project_name}/, artifacts in /mnt/artifacts/
- DFS projects: Data in /domino/datasets/local/{project_name}/, artifacts in /mnt/

Usage:
    from scripts.data_config import get_data_paths
    paths = get_data_paths('my_project')
    data_dir = paths['base_data_path']
    artifacts_dir = paths['artifacts_path']
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_data_paths(project_name: str = 'diesel_fuel_quality'):
    """
    Get appropriate data storage paths based on Domino project type.
    
    Args:
        project_name (str): Name of the project for organizing data
        
    Returns:
        dict: Dictionary containing path information
            - base_data_path: Where to store datasets
            - artifacts_path: Where to store models/reports
            - project_type: 'git' or 'dfs'
            - raw_data_path: Where to store raw/external data
            - processed_data_path: Where to store processed data
            - models_path: Where to store trained models
            - reports_path: Where to store analysis reports
    """
    
    # Determine project type based on DOMINO_WORKING_DIR
    working_dir = os.environ.get('DOMINO_WORKING_DIR', '/mnt/code')
    datasets_dir = os.environ.get('DOMINO_DATASETS_DIR', '/mnt/data')
    
    if working_dir == '/mnt/code':
        # Git-based project
        project_type = 'git'
        # Check if we have access to external data mount
        try:
            test_path = Path(datasets_dir) / project_name
            test_path.mkdir(parents=True, exist_ok=True)
            base_data_path = test_path
        except PermissionError:
            logger.warning(f"No write access to {datasets_dir}. Using /mnt/code/data/ instead.")
            base_data_path = Path('/mnt/code/data') / project_name
        
        artifacts_path = Path('/mnt/artifacts')
            
    else:
        # DFS project  
        project_type = 'dfs'
        base_data_path = Path('/domino/datasets/local') / project_name
        artifacts_path = Path('/mnt')
    
    # Create subdirectories
    raw_data_path = base_data_path / 'raw'
    processed_data_path = base_data_path / 'processed' 
    models_path = artifacts_path / 'models'
    reports_path = artifacts_path / 'reports'
    visualizations_path = artifacts_path / 'visualizations'
    
    paths = {
        'base_data_path': base_data_path,
        'artifacts_path': artifacts_path,
        'project_type': project_type,
        'raw_data_path': raw_data_path,
        'processed_data_path': processed_data_path,
        'models_path': models_path,
        'reports_path': reports_path,
        'visualizations_path': visualizations_path
    }
    
    # Create directories if they don't exist
    for path_name, path_obj in paths.items():
        if path_name not in ['project_type']:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path_obj}")
            except Exception as e:
                logger.error(f"Failed to create directory {path_obj}: {e}")
    
    # Log configuration
    logger.info(f"Project type: {project_type}")
    logger.info(f"Base data path: {base_data_path}")
    logger.info(f"Artifacts path: {artifacts_path}")
    
    return paths

def setup_project_structure(project_name: str = 'diesel_fuel_quality'):
    """
    Set up complete project structure following ML engineering best practices.
    
    Args:
        project_name (str): Name of the project
        
    Returns:
        dict: Dictionary of all created paths
    """
    
    # Get data paths
    paths = get_data_paths(project_name)
    
    # Create source code structure
    src_paths = {
        'src_api': Path('/mnt/code/src/api'),
        'src_models': Path('/mnt/code/src/models'), 
        'src_monitoring': Path('/mnt/code/src/monitoring'),
        'src_data': Path('/mnt/code/src/data'),
        'notebooks_exploration': Path('/mnt/code/notebooks/01_data_exploration'),
        'notebooks_development': Path('/mnt/code/notebooks/02_model_development'),
        'notebooks_evaluation': Path('/mnt/code/notebooks/03_model_evaluation'),
        'notebooks_deployment': Path('/mnt/code/notebooks/04_deployment_prep'),
        'config': Path('/mnt/code/config'),
        'tests_unit': Path('/mnt/code/tests/unit'),
        'tests_integration': Path('/mnt/code/tests/integration'),
        'docs_business': Path('/mnt/code/docs/business-analysis')
    }
    
    # Create all directories
    all_paths = {**paths, **src_paths}
    for path_name, path_obj in all_paths.items():
        if path_name not in ['project_type']:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path_obj}")
            except Exception as e:
                logger.error(f"Failed to create directory {path_obj}: {e}")
    
    return all_paths

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Setting up project structure for diesel fuel quality prediction...")
    paths = setup_project_structure('diesel_fuel_quality')
    
    print("\nProject structure created:")
    for name, path in paths.items():
        if name != 'project_type':
            print(f"  {name}: {path}")
    
    print(f"\nProject type: {paths['project_type']}")