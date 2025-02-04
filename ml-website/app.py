# File: ml_comparison_web/app.py

from flask import Flask, render_template
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Routes
@app.route('/')
@app.route('/home')
def home():
    """Home page route"""
    return render_template('home.html', title='Home')

@app.route('/comparison')
def comparison():
    """Model comparison page route"""
    # Mock data for demonstration - replace with actual model data
    models = [
        {
            'name': 'Random Forest',
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.86,
            'f1_score': 0.84,
            'training_time': 45.2,
            'parameters': 'n_estimators=100, max_depth=10',
            'dataset_size': '10,000 samples'
        },
        {
            'name': 'XGBoost',
            'accuracy': 0.88,
            'precision': 0.87,
            'recall': 0.89,
            'f1_score': 0.88,
            'training_time': 62.5,
            'parameters': 'n_estimators=200, learning_rate=0.1',
            'dataset_size': '10,000 samples'
        },
        {
            'name': 'Neural Network',
            'accuracy': 0.82,
            'precision': 0.81,
            'recall': 0.83,
            'f1_score': 0.82,
            'training_time': 128.7,
            'parameters': 'layers=[64,32], activation=relu',
            'dataset_size': '10,000 samples'
        }
    ]
    
    return render_template('comparison.html', 
                         title='Model Comparison',
                         models=models)

if __name__ == '__main__':
    app.run(debug=True)