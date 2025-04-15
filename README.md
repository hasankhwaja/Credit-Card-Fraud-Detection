# Credit-Card-Fraud-Detection

## Project Purpose 

The Credit Card Fraud Detection System aims to develop a solution to identify fradulent transactions using a dataset detailing the use of credit cards. The importance of this project is illustrated by the significant financial risk fradulent activities pose to credit card users, merchants as well as other financial institutions, making the need to detect and prevent fraud critical. 

This project uses a simulated dataset taken from Kaggle which was generated using the Python library Faker. This dataset reflects the real world, detailing its patterns and challenges, including features related to transactions, customers, and merchants. This collection of features allows for the creation and evaluation of a variety of machine learning models. 

## Goals

1.) Develop a scalable and real-time simulation 
* Create a real-time transaction processing system using Apache Kafka and Spark
* Use the simulated data to reflect real world scenarios

2.) Develop a Predictive Model 
* Build and train machine learning models in order to predict fraud
* Models may include, but are not limited to, Naive Bayes, Logistic Regression, and Random Forests

3.) Feature Importance Analysis
* Utilize XGBoost to evaluate and identify key identifiers of fraudulent behaviors
* Calculate feature importance metrics 

## Naive Bayes Implementation

We have now added a dedicated Naive Bayes classifier as part of this project. Below is a brief overview of the changes and how to use them:

1. New File 
   - A file named `naive_bayes.py` (or similarly named) demonstrates how we preprocess our data and train a **Gaussian Naive Bayes** classifier.  
   - It includes a function `naive_bayes_example()` that does the following:
     1. Loads and preprocesses `fraudTrain.csv`  
     2. Splits the data into training and validation sets (80/20)  
     3. Trains the Naive Bayes model and prints metrics (Accuracy, Precision, Recall, F1)  
     4. Loads and preprocesses `fraudTest.csv`  
     5. Evaluates the same model on the test data, printing final metrics  

2. Usage Instructions  
   - To run the Naive Bayes script:
     ```bash
     python naive_bayes.py
     ```
     Make sure your `fraudTrain.csv` and `fraudTest.csv` files are in the correct location (or adjust file paths if needed).  

3. Model Performance
   - In our initial runs, the Naive Bayes model achieved high overall accuracy but showed moderate precision and recall, reflecting the class imbalance typically found in fraud detection scenarios.  
   - Future enhancements (e.g., threshold tuning, oversampling with SMOTE, feature engineering) can further boost performance.  

By adding the Naive Bayes model, we aim to complement existing methods (e.g., Random Forest, and XGBoost) and compare how each performs under severe class imbalance.

## Random Forest Classifier 

We addad a random forest classifer as a part of this project. Below is a brief overview.
   - Preprocesses data for both training and test sets
   - Includes feature importance
   - Fine tuned hyperparameters
   - Best estimator randomized search
   - 3-Fold cross validation over 15 iterations
   - Random oversampling to reduce data imbalances

Results
   Accuracy - 0.997 
   Precision - 0.684
   Recall - 0.687 
   F1-Score - 0.685

Model Performance
   - The random forest classifier achieved high accuracy like all other tested models as well as balanced results across both the precision and recall metrics
   - Oversampling techniques have been shown to increase overall F-1 score

---
## Application

A web-based dashboard for monitoring and analyzing financial transactions in real-time, with machine learning-powered fraud detection capabilities.

### Overview

This application simulates a real-time transaction monitoring system that:

1. Generates realistic transaction data continuously
2. Stores transactions in a SQLite database
3. Applies machine learning models to detect potential fraud
4. Provides interactive visualizations and analysis tools
5. Logs detected fraud cases for further investigation

### Features

#### Real-time Transaction Monitoring

- Live transaction feed with fraud predictions
- Auto-refresh functionality with configurable intervals (5s, 10s, 30s, 1m)
- Visual status indicators showing system activity
- Transaction timestamps and analysis times

#### Database Statistics

- Total transaction count
- Processed vs. pending transactions
- Average transaction amount
- Total transaction volume

### Data Visualization

- **Transaction Amount Distribution**: Bar chart showing the distribution of transaction amounts across different ranges
- **Fraud Detection Rate**: Doughnut chart displaying the proportion of legitimate vs. fraudulent transactions

#### Transaction Data Explorer

Interactive SQL query interface for exploring transaction data:

- Custom SQL query execution
- Pre-built query templates for common analyses:
  - Recent Transactions
  - Category Summary
  - High-Value Transactions (>$200)
  - Top Merchants

#### Fraud Analysis

Specialized interface for analyzing detected fraud:

- Custom SQL queries against the fraud log
- Pre-built fraud analysis templates:
  - All Fraud Cases
  - Fraud by Category
  - Fraud by Merchant
  - Highest Value Fraud
- Interactive map integration for fraud locations

### Technical Implementation

#### Backend Components

- **Flask Application**: Handles HTTP requests and serves the web interface
- **Transaction Simulator**: Generates realistic transaction data
- **SQLite Database**: Stores transaction data
- **Machine Learning Model**: Predicts fraud probability for each transaction
- **Fraud Logger**: Records detected fraud cases to a JSON file

#### Frontend Components

- **Bootstrap UI**: Responsive design that works on various devices
- **Chart.js**: Interactive data visualizations
- **Real-time Updates**: AJAX-based data refreshing
- **SQL Query Interface**: Custom data exploration

### Getting Started

#### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, scikit-learn, flask, joblib

#### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn flask joblib
   ```

#### Running the Application

1. Start the Flask server:
   ```bash
   python web_app/app.py
   ```
2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

### Usage Examples

#### Exploring Transaction Data

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

#### Analyzing Fraud Patterns

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

### Architecture

The system follows a client-server architecture:

1. **Transaction Generation**: Background thread continuously generates new transactions
2. **Database Storage**: Transactions are stored in SQLite with processed/unprocessed flags
3. **Fraud Detection**: Machine learning model evaluates each transaction
4. **Fraud Logging**: Detected fraud is saved to a separate JSON file
5. **Web Interface**: Flask serves the UI and handles API requests
6. **Real-time Updates**: Client-side JavaScript polls for updates

### Future Enhancements

Potential improvements for future versions:

- User authentication and role-based access
- Email/SMS alerts for high-risk fraud cases
- Advanced anomaly detection algorithms
- Historical trend analysis
- Transaction network visualization
- Mobile app integration

## License

This project is licensed under the MIT License - see the LICENSE file for details. 





