import argparse
from expense_categorizer import ExpenseCategorizer

def main():
    """Command-line tool to categorize expenses"""
    parser = argparse.ArgumentParser(description='Categorize an expense based on description and amount')
    parser.add_argument('description', type=str, help='Description of the expense')
    parser.add_argument('amount', type=float, help='Amount of the expense')
    parser.add_argument('--vendor', type=str, help='Vendor name (optional)', default=None)
    parser.add_argument('--feedback', type=str, help='Provide correct category as feedback', default=None)
    parser.add_argument('--db', type=str, help='Path to database file', default='expenses.db')
    
    args = parser.parse_args()
    
    categorizer = ExpenseCategorizer(db_path=args.db)
    
    # Check if database exists
    if not categorizer.check_database():
        print("The expenses database doesn't exist or is not properly set up.")
        print("Please run setup_database.py first to create and populate the database.")
        return
    
    # Categorize the expense
    category, confidence = categorizer.categorize_expense(
        args.description, args.amount, args.vendor
    )
    
    print(f"\nExpense: {args.description}")
    print(f"Amount: ${args.amount:.2f}")
    if args.vendor:
        print(f"Vendor: {args.vendor}")
    print(f"Predicted category: {category}")
    print(f"Confidence: {confidence:.2f}")
    
    # If feedback is provided, update the model
    if args.feedback:
        print(f"\nReceived feedback: Correct category is '{args.feedback}'")
        categorizer.retrain_with_feedback(
            args.description, args.amount, args.feedback, args.vendor
        )
        print("Thank you for your feedback!")

if __name__ == "__main__":
    main()