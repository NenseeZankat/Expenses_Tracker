import numpy as np
import pandas as pd
import sqlite3
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ExpenseCategorizer:
    """
    A machine learning model to automatically categorize expenses based on
    transaction descriptions, amounts, and other metadata.
    """
    
    def __init__(self, db_path='expenses.db'):
        """Initialize the categorizer with path to the database."""
        self.db_path = db_path
        self.model = None
        self.model_path = 'expense_categorizer_model.pkl'
        self.categories = None
        self.min_samples_per_category = 5
        
    def check_database(self):
        """Check if the database exists and has the required tables."""
        if not os.path.exists(self.db_path):
            print(f"Database {self.db_path} not found.")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expenses';")
            table_exists = cursor.fetchone() is not None
            conn.close()
            return table_exists
        except Exception as e:
            print(f"Error checking database: {e}")
            return False
    
    def load_data(self):
        """Load expense data from SQLite database."""
        if not self.check_database():
            print("The expenses table doesn't exist. Please run setup_database.py first.")
            return pd.DataFrame()  # Return empty DataFrame
            
        conn = sqlite3.connect(self.db_path)
        # Adjust query based on your actual database schema
        query = """
        SELECT 
            description, 
            amount, 
            category, 
            vendor 
        FROM expenses
        WHERE category IS NOT NULL
        """
        try:
            expenses_df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Clean text for better feature extraction
            expenses_df['clean_description'] = expenses_df['description'].fillna('')
            expenses_df['clean_description'] = expenses_df['clean_description'].apply(
                lambda x: self._clean_text(x)
            )
            
            # Add vendor info if available
            expenses_df['vendor'] = expenses_df['vendor'].fillna('')
            expenses_df['features'] = expenses_df['clean_description'] + ' ' + expenses_df['vendor']
            
            # Store unique categories
            self.categories = expenses_df['category'].unique()
            
            print(f"Loaded {len(expenses_df)} expenses with {len(self.categories)} categories")
            return expenses_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _clean_text(self, text):
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_model(self):
        """Build and train the categorization model."""
        expenses_df = self.load_data()
        
        # Check if we have any data
        if len(expenses_df) == 0:
            print("No data available to train the model.")
            return False
            
        # Check if we have enough data
        if len(expenses_df) < 20:
            print("Not enough data to train the model. At least 20 categorized expenses needed.")
            return False
        
        # Check if each category has enough examples
        category_counts = expenses_df['category'].value_counts()
        valid_categories = category_counts[category_counts >= self.min_samples_per_category].index
        
        if len(valid_categories) < 2:
            print(f"Need at least 2 categories with {self.min_samples_per_category} examples each.")
            return False
        
        # Filter for valid categories
        expenses_df = expenses_df[expenses_df['category'].isin(valid_categories)]
        self.categories = valid_categories
        
        # Prepare features and target
        X = expenses_df['features']
        y = expenses_df['category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create a pipeline with TF-IDF and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Show detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self._save_model()
        
        return True
    
    def _save_model(self):
        """Save the trained model to disk."""
        model_data = {
            'model': self.model,
            'categories': self.categories
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a previously trained model from disk."""
        if not os.path.exists(self.model_path):
            print("No saved model found. Please train a model first.")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.categories = model_data['categories']
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def categorize_expense(self, description, amount, vendor=None):
        """
        Categorize a single expense based on its description and amount.
        
        Args:
            description (str): Expense description
            amount (float): Expense amount
            vendor (str, optional): Vendor name if available
            
        Returns:
            str: Predicted category
            float: Confidence score (probability of prediction)
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                # Fall back to a simple rule-based system if no model is available
                return self._rule_based_categorization(description, amount)
        
        # Clean and prepare the input
        clean_desc = self._clean_text(description)
        vendor = vendor or ""
        features = f"{clean_desc} {vendor}"
        
        # Get prediction and probability
        predicted_category = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        max_prob_index = np.argmax(probabilities)
        confidence = probabilities[max_prob_index]
        
        return predicted_category, confidence
    
    def _rule_based_categorization(self, description, amount):
        """
        Simple rule-based categorization as fallback when no model is available.
        
        Returns:
            str: Estimated category
            float: Confidence (fixed at 0.5 for rule-based)
        """
        description = description.lower()
        
        # Define some simple rules (customize based on your categories)
        rules = {
            'groceries': ['grocery', 'supermarket', 'food', 'market'],
            'dining': ['restaurant', 'cafe', 'coffee', 'dining'],
            'utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'bill'],
            'transportation': ['uber', 'lyft', 'taxi', 'bus', 'train', 'gas', 'fuel'],
            'entertainment': ['movie', 'cinema', 'theater', 'concert', 'netflix', 'spotify'],
            'shopping': ['amazon', 'walmart', 'target', 'store', 'buy'],
        }
        
        for category, keywords in rules.items():
            for keyword in keywords:
                if keyword in description:
                    return category, 0.5
        
        # Default category
        return 'miscellaneous', 0.3
    
    def retrain_with_feedback(self, description, amount, actual_category, vendor=None):
        """
        Update the model with user feedback by adding the corrected example
        to a feedback dataset and retraining the model periodically.
        
        Args:
            description (str): Expense description
            amount (float): Expense amount
            actual_category (str): The correct category provided by user
            vendor (str, optional): Vendor name if available
        """
        # Check if database exists
        if not self.check_database():
            print("Database not found or not properly set up. Please run setup_database.py first.")
            return
        
        # Store feedback
        conn = sqlite3.connect(self.db_path)
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
        
        # Store feedback
        cursor.execute('''
        INSERT INTO categorization_feedback 
        (description, amount, vendor, correct_category)
        VALUES (?, ?, ?, ?)
        ''', (description, amount, vendor or '', actual_category))
        
        # Also add to the main expenses table for future model training
        cursor.execute('''
        INSERT INTO expenses 
        (description, amount, vendor, category)
        VALUES (?, ?, ?, ?)
        ''', (description, amount, vendor or '', actual_category))
        
        conn.commit()
        conn.close()
        
        print(f"Feedback recorded: '{description}' categorized as '{actual_category}'")
        
        # Check if we should retrain the model
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM categorization_feedback')
        feedback_count = cursor.fetchone()[0]
        conn.close()
        
        # Retrain after collecting 10 new feedback entries
        if feedback_count % 10 == 0:
            print("Retraining model with new feedback...")
            self.build_model()

def demo():
    """Demo function to show how to use the ExpenseCategorizer."""
    categorizer = ExpenseCategorizer('expenses.db')
    
    # Check if database exists and is properly set up
    if not categorizer.check_database():
        print("The expenses database doesn't exist or is not properly set up.")
        print("Please run setup_database.py first to create and populate the database.")
        return
    
    # Try to load existing model, build if none exists
    if not categorizer.load_model():
        print("Building new model...")
        success = categorizer.build_model()
        if not success:
            print("Could not build model. Using rule-based categorization.")
    
    # Demo categorization
    test_expenses = [
        {"description": "Walmart Grocery Shopping", "amount": 85.47},
        {"description": "Shell Gas Station", "amount": 45.00},
        {"description": "Netflix Monthly Subscription", "amount": 13.99},
        {"description": "Starbucks Coffee", "amount": 5.75},
        {"description": "AT&T Phone Bill", "amount": 75.00},
        {"description": "Unknown Transaction", "amount": 42.50}  # Test for something not obvious
    ]
    
    print("\nCategorizing test expenses:")
    for expense in test_expenses:
        category, confidence = categorizer.categorize_expense(
            expense["description"], expense["amount"]
        )
        print(f"{expense['description']} (${expense['amount']:.2f}): {category} (confidence: {confidence:.2f})")
    
    # Demo feedback
    print("\nProviding feedback to improve the model:")
    categorizer.retrain_with_feedback(
        "Uber Ride to Airport", 35.00, "transportation"
    )

if __name__ == "__main__":
    demo()