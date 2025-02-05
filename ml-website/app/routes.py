from flask import Blueprint, render_template, jsonify
from app.models.ml_models import MLModelsComparison

main = Blueprint('main', __name__)

@main.route('/')
@main.route('/home')
def home():
    """หน้าแรกของเว็บไซต์"""
    return render_template('home.html', title='Home')

@main.route('/comparison')
def comparison():
    """หน้าแสดงการเปรียบเทียบโมเดล"""
    try:
        ml_comparison = MLModelsComparison()
        models_data = ml_comparison.get_models_comparison()
        return render_template('comparison.html', 
                             title='Model Comparison',
                             models=models_data)
    except Exception as e:
        print(f"Error loading model comparison: {str(e)}")
        return render_template('comparison.html', 
                             title='Model Comparison',
                             error="Unable to load model comparison data")

@main.route('/api/model-metrics')
def model_metrics():
    """API endpoint สำหรับข้อมูลเมทริกซ์ของโมเดล"""
    try:
        ml_comparison = MLModelsComparison()
        metrics_data = ml_comparison.get_detailed_metrics()
        return jsonify(metrics_data)
    except Exception as e:
        print(f"Error fetching model metrics: {str(e)}")
        return jsonify({
            'error': 'Unable to load model metrics data'
        }), 500  # HTTP 500 Internal Server Error