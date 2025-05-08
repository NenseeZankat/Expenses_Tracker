import sqlite3
import os

def setup_expenses_database(db_path='expenses.db'):
    """
    Create and initialize the expenses database with sample data.
    """
    # Check if database file already exists
    if os.path.exists(db_path):
        print(f"Database {db_path} already exists. Do you want to reset it? (y/n)")
        response = input().lower().strip()
        if response != 'y':
            print("Database setup cancelled.")
            return False
        
    # Create or connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create expenses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY,
        description TEXT NOT NULL,
        amount REAL NOT NULL,
        category TEXT,
        vendor TEXT,
        date TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create feedback table for model improvements
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
    
    # Sample expense data
    sample_expenses = [
        ("Walmart Grocery Shopping", 85.47, "groceries", "Walmart"),
        ("Trader Joe's", 67.89, "groceries", "Trader Joe's"),
        ("Kroger", 112.34, "groceries", "Kroger"),
        ("Safeway", 94.52, "groceries", "Safeway"),
        ("Whole Foods Market", 158.75, "groceries", "Whole Foods"),
        
        ("Shell Gas Station", 45.00, "transportation", "Shell"),
        ("Exxon", 52.35, "transportation", "Exxon"),
        ("Chevron", 48.75, "transportation", "Chevron"),
        ("BP Gas", 50.25, "transportation", "BP"),
        ("Uber Ride", 24.50, "transportation", "Uber"),
        ("Lyft", 18.75, "transportation", "Lyft"),
        
        ("Netflix Monthly Subscription", 13.99, "entertainment", "Netflix"),
        ("Spotify Premium", 9.99, "entertainment", "Spotify"),
        ("AMC Movie Tickets", 28.50, "entertainment", "AMC Theaters"),
        ("Disney+ Subscription", 7.99, "entertainment", "Disney"),
        ("HBO Max", 14.99, "entertainment", "HBO"),
        
        ("Starbucks Coffee", 5.75, "dining", "Starbucks"),
        ("Chipotle", 12.50, "dining", "Chipotle"),
        ("Olive Garden", 45.82, "dining", "Olive Garden"),
        ("McDonald's", 8.45, "dining", "McDonald's"),
        ("Panera Bread", 15.67, "dining", "Panera"),
        ("Local Coffee Shop", 4.25, "dining", "Local Cafe"),
        
        ("AT&T Phone Bill", 75.00, "utilities", "AT&T"),
        ("Electric Company Bill", 120.45, "utilities", "Electric Co."),
        ("Water Bill", 45.30, "utilities", "Water Utility"),
        ("Internet Service", 65.99, "utilities", "Comcast"),
        ("Natural Gas Bill", 35.25, "utilities", "Gas Company"),
        
        ("Amazon.com Purchase", 34.67, "shopping", "Amazon"),
        ("Target", 87.34, "shopping", "Target"),
        ("Best Buy", 199.99, "shopping", "Best Buy"),
        ("Home Depot", 145.28, "shopping", "Home Depot"),
        ("Macy's", 158.75, "shopping", "Macy's"),
        
        ("Pharmacy", 28.45, "health", "CVS Pharmacy"),
        ("Doctor Visit Copay", 25.00, "health", "Medical Center"),
        ("Gym Membership", 45.00, "health", "Fitness Club"),
        ("Dental Checkup", 75.00, "health", "Dental Office"),
        
        ("Monthly Rent", 1200.00, "housing", "Landlord"),
        ("Home Insurance", 95.00, "housing", "Insurance Co."),
        ("HOA Fee", 250.00, "housing", "HOA"),
        
        ("Office Supplies", 42.99, "work", "Office Depot"),
        ("Work Lunch Meeting", 85.34, "work", "Restaurant"),
        ("Professional Subscription", 15.00, "work", "Professional Org"),
        
        ("Dog Food", 32.99, "pets", "Pet Store"),
        ("Veterinary Visit", 85.00, "pets", "Vet Clinic"),
        ("Pet Supplies", 28.45, "pets", "PetSmart"),
        
        ("Donation", 50.00, "charity", "Red Cross"),
        ("School Fundraiser", 25.00, "charity", "Local School"),
        
        ("Birthday Gift", 45.00, "gifts", "Department Store"),
        ("Anniversary Present", 125.00, "gifts", "Jewelry Store"),
        
        ("Laptop Repair", 150.00, "miscellaneous", "Tech Shop"),
        ("Parking Fee", 15.00, "miscellaneous", "Parking Garage"),
        ("ATM Withdrawal", 100.00, "miscellaneous", "Bank")
    ]
    
    # Insert sample expenses
    cursor.executemany('''
    INSERT INTO expenses (description, amount, category, vendor)
    VALUES (?, ?, ?, ?)
    ''', sample_expenses)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database {db_path} has been created and populated with {len(sample_expenses)} sample expenses.")
    print("You can now run the expense categorizer model.")
    return True

if __name__ == "__main__":
    setup_expenses_database()