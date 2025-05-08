# Expense Tracker - MERN Stack Application

A full-stack expense tracking application with machine learning-powered expense categorization. Built with MongoDB, Express, React, and Node.js.

## Prerequisites

- Node.js (v14.0+)
- npm or yarn
- MongoDB (local or Atlas)
- Python 3.8+ (for ML model)
- Required Python packages (`numpy`, `pandas`, `scikit-learn`, `sqlite3`)

## Installation & Setup

### 1. Clone the repository

bash
git clone https://github.com/yourusername/expenses-tracker.git
cd expenses-tracker


### 2. Setup Backend

bash
# Navigate to the server directory
cd server

# Install dependencies
npm install

# Create a .env file with your configuration
echo "PORT=5000
MONGODB_URI=mongodb://localhost:27017/expenses-tracker
JWT_SECRET=your_jwt_secret_key
NODE_ENV=development" > .env

### 3. Setup Frontend

bash
# Navigate to the client directory
cd ../client

# Install dependencies
npm install

### 4. Setup ML Model

bash
# Navigate to the model directory
cd ../model

# Install required Python packages
pip install numpy pandas scikit-learn

# Initialize the database with sample data
python setup_database.py

# Build the ML model
python expense_categorizer.py

## Running the Application

### 1. Start the Backend Server

bash
# From the server directory
cd server
npm run dev

The server will run on http://localhost:5000 by default.

### 2. Start the Frontend Development Server

bash
# From the client directory
cd client
npm start


The React application will run on http://localhost:3000 by default.

### 3. Using the ML Model

You can use the ML model separately via command line:

bash
# From the model directory
cd model
python categorize_expense.py "Grocery shopping at Trader Joe's" 75.32 --vendor "Trader Joe's"


## Production Deployment

### Build the Frontend

bash
# From the client directory
cd client
npm run build


### Start Production Server

bash
# From the server directory
cd server
NODE_ENV=production npm start

## Database Configuration

### MongoDB

The application uses MongoDB to store user data, expenses, and categories. You can use either a local MongoDB instance or MongoDB Atlas.

### SQLite (for ML Model)

The ML model uses SQLite to store training data and feedback. The database file is located at `model/expenses.db`.

## Available Scripts

### Backend

bash
# Run development server with nodemon
npm run dev

# Run production server
npm start


### Frontend

bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test


## ML Model Commands

bash
# Initialize database with sample data
python setup_database.py

# Train the model and run demo
python expense_categorizer.py

# Categorize a single expense
python categorize_expense.py "description" amount --vendor "vendor_name"

# Categorize with feedback
python categorize_expense.py "description" amount --vendor "vendor_name" --feedback "correct_category"


## Environment Variables

### Backend (.env file in server directory)
- `PORT` - Server port (default: 5000)
- `MONGODB_URI` - MongoDB connection string
- `JWT_SECRET` - Secret for JWT token generation
- `NODE_ENV` - Environment (development, production)

### Frontend (.env file in client directory)
- `REACT_APP_API_URL` - Backend API URL (default: http://localhost:5000/api)

## Integrating the ML Model with Node.js

The ML model is integrated with the Node.js backend using the `child_process` module to execute Python scripts. 
