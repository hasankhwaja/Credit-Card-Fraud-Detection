# Real-time Fraud Detection System

A web-based dashboard for monitoring and analyzing financial transactions in real-time, with machine learning-powered fraud detection capabilities.

## Overview

This application simulates a real-time transaction monitoring system that:

1. Generates realistic transaction data continuously
2. Stores transactions in a SQLite database
3. Applies machine learning models to detect potential fraud
4. Provides interactive visualizations and analysis tools
5. Logs detected fraud cases for further investigation

## Features

### Real-time Transaction Monitoring

- Live transaction feed with fraud predictions
- Auto-refresh functionality with configurable intervals (5s, 10s, 30s, 1m)
- Visual status indicators showing system activity
- Transaction timestamps and analysis times

### Database Statistics

- Total transaction count
- Processed vs. pending transactions
- Average transaction amount
- Total transaction volume

### Data Visualization

- **Transaction Amount Distribution**: Bar chart showing the distribution of transaction amounts across different ranges
- **Fraud Detection Rate**: Doughnut chart displaying the proportion of legitimate vs. fraudulent transactions

### Transaction Data Explorer

Interactive SQL query interface for exploring transaction data:

- Custom SQL query execution
- Pre-built query templates for common analyses:
  - Recent Transactions
  - Category Summary
  - High-Value Transactions (>$200)
  - Top Merchants

### Fraud Analysis

Specialized interface for analyzing detected fraud:

- Custom SQL queries against the fraud log
- Pre-built fraud analysis templates:
  - All Fraud Cases
  - Fraud by Category
  - Fraud by Merchant
  - Highest Value Fraud
- Interactive map integration for fraud locations

## Technical Implementation

### Backend Components

- **Flask Application**: Handles HTTP requests and serves the web interface
- **Transaction Simulator**: Generates realistic transaction data
- **SQLite Database**: Stores transaction data
- **Machine Learning Model**: Predicts fraud probability for each transaction
- **Fraud Logger**: Records detected fraud cases to a JSON file

### Frontend Components

- **Bootstrap UI**: Responsive design that works on various devices
- **Chart.js**: Interactive data visualizations
- **Real-time Updates**: AJAX-based data refreshing
- **SQL Query Interface**: Custom data exploration

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, flask, joblib

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn flask joblib
   ```

### Running the Application

1. Start the Flask server:
   ```bash
   python web_app/app.py
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Examples

### Exploring Transaction Data

1. Navigate to the "Transaction Data Explorer" section
2. Select a pre-built query or write your own SQL
3. Click "Execute Query" to see results

Example queries:
```sql
-- Find high-value transactions
SELECT * FROM transactions WHERE amount > 200 ORDER BY amount DESC

-- Analyze transactions by category
SELECT category, COUNT(*) as count, AVG(amount) as avg_amount 
FROM transactions 
GROUP BY category 
ORDER BY count DESC
```

### Analyzing Fraud Patterns

1. Navigate to the "Fraud Analysis" section
2. Select a pre-built query or write your own
3. Click "Execute Query" to analyze fraud data

Example queries:
```sql
-- View all fraud cases
SELECT * FROM fraud_transactions ORDER BY timestamp DESC

-- Analyze fraud by merchant
SELECT merchant, COUNT(*) as count 
FROM fraud_transactions 
GROUP BY merchant 
ORDER BY count DESC
```

## Architecture

The system follows a client-server architecture:

1. **Transaction Generation**: Background thread continuously generates new transactions
2. **Database Storage**: Transactions are stored in SQLite with processed/unprocessed flags
3. **Fraud Detection**: Machine learning model evaluates each transaction
4. **Fraud Logging**: Detected fraud is saved to a separate JSON file
5. **Web Interface**: Flask serves the UI and handles API requests
6. **Real-time Updates**: Client-side JavaScript polls for updates

## Future Enhancements

Potential improvements for future versions:

- User authentication and role-based access
- Email/SMS alerts for high-risk fraud cases
- Advanced anomaly detection algorithms
- Historical trend analysis
- Transaction network visualization
- Mobile app integration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 