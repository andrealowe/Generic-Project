"""
Enterprise Dashboard Configuration

Professional styling and configuration settings for the Diesel Quality Management Dashboard.
Designed for Oil & Gas industry presentations and executive reporting.
"""

# Corporate Color Palette
CORPORATE_COLORS = {
    'primary_blue': '#1E3A8A',      # Deep corporate blue
    'secondary_blue': '#3B82F6',    # Medium blue  
    'accent_blue': '#60A5FA',       # Light blue
    'success_green': '#059669',     # Success indicator
    'warning_orange': '#D97706',    # Warning indicator
    'danger_red': '#DC2626',        # Critical indicator
    'neutral_gray': '#6B7280',      # Neutral text
    'light_gray': '#F3F4F6',        # Background
    'white': '#FFFFFF',             # Pure white
    'dark_gray': '#374151'          # Dark text
}

# ASTM D975 Compliance Standards
ASTM_D975_STANDARDS = {
    'cetane_number': {
        'min': 40, 
        'target': 45, 
        'excellent': 50,
        'description': 'Ignition quality indicator'
    },
    'sulfur_ppm': {
        'max': 15, 
        'target': 10, 
        'excellent': 5,
        'description': 'Ultra Low Sulfur Diesel requirement'
    },
    'viscosity_cst': {
        'min': 1.9, 
        'max': 4.1, 
        'target': 3.0,
        'description': 'Kinematic viscosity at 40°C'
    },
    'flash_point_c': {
        'min': 52, 
        'target': 60, 
        'excellent': 70,
        'description': 'Safety parameter for handling'
    },
    'density_gcc': {
        'min': 0.82, 
        'max': 0.845, 
        'target': 0.835,
        'description': 'Density at 15°C'
    },
    'cloud_point_c': {
        'max': -12, 
        'target': -15, 
        'excellent': -20,
        'description': 'Cold weather performance'
    },
    'aromatic_content': {
        'max': 35, 
        'target': 25, 
        'excellent': 15,
        'description': 'Aromatic hydrocarbon percentage'
    },
    'quality_score': {
        'min': 70, 
        'target': 85, 
        'excellent': 95,
        'description': 'Composite quality assessment'
    }
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'title': 'Enterprise Diesel Quality Management',
    'subtitle': 'Real-time monitoring and compliance tracking for refinery operations',
    'version': '2.0',
    'presentation_mode': True,
    'large_fonts': True,
    'high_contrast': True
}

# Chart Styling Configurations
CHART_STYLE = {
    'font_family': 'Inter',
    'title_size': 16,
    'axis_title_size': 14,
    'tick_size': 12,
    'legend_size': 11,
    'background_color': 'rgba(0,0,0,0)',
    'grid_color': 'rgba(0,0,0,0.1)',
    'line_width': 3,
    'marker_size': 6
}

# Professional Layout Settings
LAYOUT_CONFIG = {
    'sidebar_width': 350,
    'main_content_padding': '2rem',
    'section_spacing': '2rem',
    'card_shadow': '0 2px 4px rgba(0,0,0,0.1)',
    'border_radius': '8px',
    'header_height': '120px'
}

def get_professional_theme():
    """Return complete professional theme configuration."""
    return {
        'colors': CORPORATE_COLORS,
        'standards': ASTM_D975_STANDARDS,
        'dashboard': DASHBOARD_CONFIG,
        'charts': CHART_STYLE,
        'layout': LAYOUT_CONFIG
    }

def get_gauge_config(parameter, value, standards):
    """Get professional gauge configuration for a specific parameter."""
    base_config = {
        'height': 400,
        'font_family': CHART_STYLE['font_family'],
        'title_size': CHART_STYLE['title_size'],
        'background': CHART_STYLE['background_color']
    }
    
    if parameter == 'cetane_number':
        base_config.update({
            'range_max': 60,
            'steps': [
                {'range': [0, 40], 'color': CORPORATE_COLORS['danger_red']},
                {'range': [40, 45], 'color': CORPORATE_COLORS['warning_orange']},
                {'range': [45, 50], 'color': '#10B981'},
                {'range': [50, 60], 'color': CORPORATE_COLORS['success_green']}
            ],
            'threshold': 40
        })
    elif parameter == 'sulfur_ppm':
        base_config.update({
            'range_max': 500,
            'steps': [
                {'range': [0, 5], 'color': CORPORATE_COLORS['success_green']},
                {'range': [5, 10], 'color': '#10B981'},
                {'range': [10, 15], 'color': CORPORATE_COLORS['warning_orange']},
                {'range': [15, 500], 'color': CORPORATE_COLORS['danger_red']}
            ],
            'threshold': 15
        })
    elif parameter == 'viscosity_cst':
        base_config.update({
            'range_max': 5,
            'steps': [
                {'range': [1, 1.9], 'color': CORPORATE_COLORS['danger_red']},
                {'range': [1.9, 4.1], 'color': CORPORATE_COLORS['success_green']},
                {'range': [4.1, 5], 'color': CORPORATE_COLORS['danger_red']}
            ],
            'threshold': 3.0
        })
    else:
        # Default configuration
        range_max = max(value * 1.2, 100)
        base_config.update({
            'range_max': range_max,
            'steps': [
                {'range': [0, range_max * 0.7], 'color': CORPORATE_COLORS['success_green']},
                {'range': [range_max * 0.7, range_max], 'color': CORPORATE_COLORS['warning_orange']}
            ],
            'threshold': standards.get('target', value)
        })
    
    return base_config

def get_kpi_styling(value, target, excellent=None):
    """Get KPI card styling based on performance thresholds."""
    if excellent and value >= excellent:
        return {
            'color': CORPORATE_COLORS['success_green'],
            'icon': '⭐',
            'status': 'Excellent'
        }
    elif value >= target:
        return {
            'color': CORPORATE_COLORS['success_green'],
            'icon': '✅',
            'status': 'Good'
        }
    elif value >= target * 0.9:
        return {
            'color': CORPORATE_COLORS['warning_orange'],
            'icon': '⚠️',
            'status': 'Warning'
        }
    else:
        return {
            'color': CORPORATE_COLORS['danger_red'],
            'icon': '❌',
            'status': 'Critical'
        }

# Export utility functions
__all__ = [
    'CORPORATE_COLORS',
    'ASTM_D975_STANDARDS', 
    'DASHBOARD_CONFIG',
    'CHART_STYLE',
    'LAYOUT_CONFIG',
    'get_professional_theme',
    'get_gauge_config',
    'get_kpi_styling'
]