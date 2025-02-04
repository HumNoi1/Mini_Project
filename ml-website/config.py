# File: ml-website/config.py

import os

class Config:
    """Flask application configuration"""
    SECRET_KEY = 'your-secret-key-here'
    STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    TEMPLATES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')