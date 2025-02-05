from flask import Flask
from config import Config
import os

def create_app(config_class=Config):
    # กำหนด template_folder และ static_folder ให้ถูกต้อง
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)
    
    app.config.from_object(config_class)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    # สร้างโฟลเดอร์ที่จำเป็น
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    return app