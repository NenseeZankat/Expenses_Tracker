from flask import request, jsonify, Blueprint
from expense_categorizer import ExpenseCategorizer
import sqlite3
import json

# Create a Blueprint for AI features
ai_features = Blueprint('ai_features', __name__)

# Initialize the categorizer
categorizer = ExpenseCategorizer('expenses.db')
categorizer.load_model()  # Try to load existing model

@ai_features.route('/categorize', methods=['POST'])
def categorize_expense():
    """
    API endpoint to categorize an expense
    
    Expected JSON payload:
    {
        "description": "Grocery shopping at Whole Foods",
        "amount": 87.65,
        "vendor": "Whole Foods" (optional)
    }
    
    Returns:
    {
        "category": "groceries",
        "confidence": 0.92
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'description' not in data or 'amount' not in data:
            return jsonify({
                'error': 'Missing required fields: description and amount'
            }), 400
            
        description = data['description']
        amount = float(data['amount'])
        vendor = data.get('vendor', None)
        
        category, confidence = categorizer.categorize_expense(description, amount, vendor)
        
        return jsonify({
            'category': category,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_features.route('/feedback', methods=['POST'])
def provide_feedback():
    """
    API endpoint for users to provide feedback on categorizations
    
    Expected JSON payload:
    {
        "description": "Grocery shopping at Whole Foods",
        "amount": 87.65,
        "vendor": "Whole Foods" (optional),
        "correct_category": "groceries"
    }
    
    Returns:
    {
        "success": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'description' not in data or 'amount' not in data or 'correct_category' not in data:
            return jsonify({
                'error': 'Missing required fields: description, amount, and correct_category'
            }), 400
            
        description = data['description']
        amount = float(data['amount'])
        correct_category = data['correct_category']
        vendor = data.get('vendor', None)
        
        categorizer.retrain_with_feedback(description, amount, correct_category, vendor)
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_features.route('/retrain', methods=['POST'])
def force_retrain():
    """
    API endpoint to force model retraining
    
    Returns:
    {
        "success": true,
        "accuracy": 0.85 (if model was successfully trained)
    }
    """
    try:
        success = categorizer.build_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Not enough data to train model'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_features.route('/categories', methods=['GET'])
def get_categories():
    """
    API endpoint to get all available categories
    
    Returns:
    {
        "categories": ["groceries", "utilities", "entertainment", ...]
    }
    """
    try:
        # Ensure the model is loaded
        if categorizer.categories is None:
            categorizer.load_model() 
            
        if categorizer.categories is None:
            # Connect to the database to get categories
            conn = sqlite3.connect('expenses.db')
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT category FROM expenses WHERE category IS NOT NULL')
            categories = [row[0] for row in cursor.fetchall()]
            conn.close()
        else:
            categories = categorizer.categories.tolist()
            
        return jsonify({
            'categories': categories
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_features.route('/stats', methods=['GET'])
def get_model_stats():
    """
    API endpoint to get statistics about the categorization model
    
    Returns:
    {
        "model_available": true,
        "categories_count": 10,
        "training_data_size": 500,
        "last_trained": "2023-08-15 14:30:22"
    }
    """
    try:
        # Check if model exists
        model_exists = categorizer.load_model()
        
        if not model_exists:
            return jsonify({
                'model_available': False
            })
            
        # Get stats from database
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        
        # Count total training examples
        cursor.execute('SELECT COUNT(*) FROM expenses WHERE category IS NOT NULL')
        training_size = cursor.fetchone()[0]
        
        # Count categories
        cursor.execute('SELECT COUNT(DISTINCT category) FROM expenses WHERE category IS NOT NULL')
        categories_count = cursor.fetchone()[0]
        
        # Get last training time
        cursor.execute('''
        SELECT timestamp FROM categorization_feedback 
        ORDER BY timestamp DESC LIMIT 1
        ''')
        result = cursor.fetchone()
        last_trained = result[0] if result else "Never"
        
        conn.close()
        
        return jsonify({
            'model_available': True,
            'categories_count': categories_count,
            'training_data_size': training_size,
            'last_trained': last_trained
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to register the blueprint with your Flask app
def register_ai_features(app):
    app.register_blueprint(ai_features, url_prefix='/api/ai')
    print("AI features registered successfully!")
    
    # Create necessary tables if they don't exist
    with app.app_context():
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        
        # Create feedback table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS categorization_feedback (
            id INTEGER PRIMARY KEY,
            description TEXT,
            amount REAL,
            vendor TEXT,
            suggested_category TEXT,
            correct_category TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()