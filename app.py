import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys
import threading
import json
from datetime import datetime
from transaction_simulator import TransactionSimulator, generate_background_transactions
import sqlite3

# Add parent directory to path to import preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now we can import preprocessing after adding parent_dir to path

app = Flask(__name__)

# Initialize the transaction simulator
simulator = TransactionSimulator(
    db_path=os.path.join(current_dir, 'transactions.db'))

# Path for fraud log file
FRAUD_LOG_FILE = os.path.join(current_dir, 'fraud_transactions.json')


def save_fraud_transaction(transaction, prediction_data):
    """Save a fraudulent transaction to the fraud log file"""
    fraud_entry = {
        'timestamp': datetime.now().isoformat(),
        'transaction_date': transaction['transaction_date'],
        'category': transaction['category'],
        'amount': transaction['amount'],
        'merchant': transaction['merchant'],
        'merchant_lat': float(transaction['merchant_lat']),
        'merchant_long': float(transaction['merchant_long']),
        'city_pop': int(transaction['city_pop']),
        'prediction_confidence': float(prediction_data['confidence']) if 'confidence' in prediction_data else None
    }

    # Load existing fraud transactions
    existing_frauds = []
    if os.path.exists(FRAUD_LOG_FILE):
        try:
            with open(FRAUD_LOG_FILE, 'r') as f:
                existing_frauds = json.load(f)
        except json.JSONDecodeError:
            existing_frauds = []

    # Add new fraud transaction
    existing_frauds.append(fraud_entry)

    # Save back to file
    with open(FRAUD_LOG_FILE, 'w') as f:
        json.dump(existing_frauds, f, indent=2)


# Start background transaction generation
transaction_thread = threading.Thread(
    target=generate_background_transactions,
    args=(simulator,),
    daemon=True
)
transaction_thread.start()


def preprocess_fraud_data(df):
    """
    Preprocess the fraud dataset using only the 8 selected features.

    Parameters:
        df (DataFrame): Input DataFrame with transaction data.

    Returns:
        tuple: Processed features (X) and the fitted scaler.
    """
    # Convert 'trans_date_trans_time' to datetime and get unix timestamp
    df['trans_date_trans_time'] = pd.to_datetime(df['transaction_date'])
    df['unix_time'] = df['trans_date_trans_time'].astype(np.int64) // 10**9

    # Calculate distance using merchant coordinates
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    # Calculate distance (using random user coordinates for simulation)
    df['user_lat'] = np.random.uniform(25, 50, size=len(df))
    df['user_long'] = np.random.uniform(-130, -70, size=len(df))
    df['distance'] = df.apply(
        lambda row: haversine(row['user_lat'], row['user_long'],
                              row['merchant_lat'], row['merchant_long']),
        axis=1
    )

    # Encode categorical columns
    categorical_columns = ['merchant', 'category', 'job']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Select features for prediction
    selected_features = ['merchant', 'category', 'amount',
                         'city_pop', 'job', 'unix_time', 'age', 'distance']
    X = df[selected_features]

    # Normalize numerical columns
    numerical_columns = ['amount', 'city_pop', 'unix_time', 'age', 'distance']
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    return X, scaler


# Load the model
model = joblib.load(os.path.join(parent_dir, 'best_rf_model.pkl'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        # Get unprocessed transactions
        transactions_df = simulator.get_unprocessed_transactions(limit=20)

        if len(transactions_df) == 0:
            return jsonify({
                'results': [],
                'total_amount': "$0.00",
                'timestamp': pd.Timestamp.now().isoformat()
            })

        # Preprocess the transactions
        X, _ = preprocess_fraud_data(transactions_df)

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(
            X)[:, 1]  # Get probability of fraud

        # Calculate total amount
        total_amount = transactions_df['amount'].sum()

        # Prepare results
        results = []
        for i, (idx, row) in enumerate(transactions_df.iterrows()):
            is_fraud = bool(predictions[i])

            # Save fraudulent transactions to file
            if is_fraud:
                save_fraud_transaction(row, {
                    'confidence': probabilities[i] * 100
                })

            results.append({
                'transaction_date': row['transaction_date'],
                'category': row['category'],
                'amount': f"${row['amount']:.2f}",
                'predicted_fraud': is_fraud
            })

        # Mark transactions as processed
        simulator.mark_transactions_as_processed(
            transactions_df['id'].tolist())

        return jsonify({
            'results': results,
            'total_amount': f"${total_amount:.2f}",
            'timestamp': pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/db_stats')
def get_db_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(simulator.db_path)
        cursor = conn.cursor()

        # Get total number of transactions
        cursor.execute('SELECT COUNT(*) FROM transactions')
        total_transactions = cursor.fetchone()[0]

        # Get number of processed transactions
        cursor.execute('SELECT COUNT(*) FROM transactions WHERE processed = 1')
        processed_transactions = cursor.fetchone()[0]

        # Get total amount of all transactions
        cursor.execute('SELECT SUM(amount) FROM transactions')
        total_amount = cursor.fetchone()[0] or 0

        # Get transaction counts by category
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM transactions 
            GROUP BY category
        ''')
        category_stats = dict(cursor.fetchall())

        # Get average transaction amount
        cursor.execute('SELECT AVG(amount) FROM transactions')
        avg_amount = cursor.fetchone()[0] or 0

        # Get amount distribution
        cursor.execute('''
            SELECT 
                CASE
                    WHEN amount <= 50 THEN '0-50'
                    WHEN amount <= 100 THEN '51-100'
                    WHEN amount <= 500 THEN '101-500'
                    WHEN amount <= 1000 THEN '501-1000'
                    ELSE '1000+'
                END as range,
                COUNT(*) as count
            FROM transactions
            GROUP BY range
            ORDER BY range
        ''')
        amount_distribution = [row[1] for row in cursor.fetchall()]

        # Get transactions per minute (last 5 minutes)
        cursor.execute('''
            SELECT COUNT(*) 
            FROM transactions 
            WHERE transaction_date >= datetime('now', '-5 minutes')
        ''')
        transactions_per_minute = (cursor.fetchone()[0] or 0) / 5

        conn.close()

        return jsonify({
            'total_transactions': total_transactions,
            'processed_transactions': processed_transactions,
            'pending_transactions': total_transactions - processed_transactions,
            'total_amount': f"${total_amount:.2f}",
            'average_amount': f"${avg_amount:.2f}",
            'category_distribution': category_stats,
            'transactions_amounts': amount_distribution,
            'transactions_per_minute': round(transactions_per_minute, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/transaction_history')
def get_transaction_history():
    """Get transaction history with optional filters"""
    try:
        # Get query parameters
        limit = request.args.get('limit', default=50, type=int)
        category = request.args.get('category', default=None, type=str)
        min_amount = request.args.get('min_amount', default=None, type=float)
        max_amount = request.args.get('max_amount', default=None, type=float)

        # Build the query
        query = 'SELECT * FROM transactions WHERE 1=1'
        params = []

        if category:
            query += ' AND category = ?'
            params.append(category)

        if min_amount is not None:
            query += ' AND amount >= ?'
            params.append(min_amount)

        if max_amount is not None:
            query += ' AND amount <= ?'
            params.append(max_amount)

        query += ' ORDER BY transaction_date DESC LIMIT ?'
        params.append(limit)

        # Execute query
        conn = sqlite3.connect(simulator.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Format the results
        transactions = []
        for _, row in df.iterrows():
            transactions.append({
                'id': row['id'],
                'date': row['transaction_date'],
                'category': row['category'],
                'amount': f"${row['amount']:.2f}",
                'merchant': row['merchant'],
                'processed': bool(row['processed'])
            })

        return jsonify({
            'transactions': transactions,
            'count': len(transactions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/category_summary')
def get_category_summary():
    """Get summary statistics by category"""
    try:
        conn = sqlite3.connect(simulator.db_path)
        cursor = conn.cursor()

        # Get summary statistics by category
        cursor.execute('''
            SELECT 
                category,
                COUNT(*) as transaction_count,
                AVG(amount) as avg_amount,
                MIN(amount) as min_amount,
                MAX(amount) as max_amount,
                SUM(amount) as total_amount
            FROM transactions
            GROUP BY category
        ''')

        categories = []
        for row in cursor.fetchall():
            categories.append({
                'category': row[0],
                'transaction_count': row[1],
                'average_amount': f"${row[2]:.2f}",
                'min_amount': f"${row[3]:.2f}",
                'max_amount': f"${row[4]:.2f}",
                'total_amount': f"${row[5]:.2f}"
            })

        conn.close()

        return jsonify({
            'categories': categories
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/fraud_log')
def get_fraud_log():
    """Get the list of detected fraudulent transactions"""
    try:
        if os.path.exists(FRAUD_LOG_FILE):
            with open(FRAUD_LOG_FILE, 'r') as f:
                fraud_transactions = json.load(f)
        else:
            fraud_transactions = []

        return jsonify({
            'fraud_transactions': fraud_transactions,
            'count': len(fraud_transactions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/execute_query', methods=['POST'])
def execute_query():
    """Execute a SQL query on the transactions database"""
    try:
        # Get the query from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        query = data['query']

        # Validate query - only allow SELECT statements for security
        if not query.strip().upper().startswith('SELECT'):
            return jsonify({'error': 'Only SELECT queries are allowed'}), 403

        # Execute the query
        conn = sqlite3.connect(simulator.db_path)

        # Convert results to dictionaries
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()

        # Convert to list of dictionaries
        results = []
        for row in rows:
            results.append({key: row[key] for key in row.keys()})

        conn.close()

        return jsonify({
            'results': results,
            'count': len(results)
        })
    except sqlite3.Error as e:
        return jsonify({'error': f'SQLite error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/execute_fraud_query', methods=['POST'])
def execute_fraud_query():
    """Execute a query on the fraud transactions data"""
    try:
        # Get the query from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        query = data['query']

        # Validate query - only allow SELECT statements for security
        if not query.strip().upper().startswith('SELECT'):
            return jsonify({'error': 'Only SELECT queries are allowed'}), 403

        # Load fraud transactions from JSON file
        if os.path.exists(FRAUD_LOG_FILE):
            with open(FRAUD_LOG_FILE, 'r') as f:
                fraud_transactions = json.load(f)
        else:
            fraud_transactions = []

        if not fraud_transactions:
            return jsonify({
                'results': [],
                'count': 0
            })

        # Convert to DataFrame for SQL-like querying
        df = pd.DataFrame(fraud_transactions)

        # Handle special case queries
        if 'fraud_transactions' in query:
            # Replace the table name with the actual DataFrame
            modified_query = query.replace(
                'FROM fraud_transactions', 'FROM df')

            try:
                # Execute the query using pandas
                result_df = pd.read_sql_query(
                    modified_query, pd.DataFrame({'df': [1]}))
                results = result_df.to_dict(orient='records')
                return jsonify({
                    'results': results,
                    'count': len(results)
                })
            except Exception as e:
                # If pandas query fails, try a more direct approach
                pass

        # Handle common queries directly
        if query == 'SELECT * FROM fraud_transactions ORDER BY timestamp DESC':
            results = df.sort_values(
                'timestamp', ascending=False).to_dict(orient='records')
        elif query == 'SELECT * FROM fraud_transactions ORDER BY amount DESC':
            # Convert amount from string to float for proper sorting
            df['amount_value'] = df['amount'].apply(lambda x: float(
                x.replace('$', '')) if isinstance(x, str) else float(x))
            results = df.sort_values('amount_value', ascending=False).drop(
                'amount_value', axis=1).to_dict(orient='records')
        elif 'GROUP BY category' in query:
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            results = category_counts.to_dict(orient='records')
        elif 'GROUP BY merchant' in query:
            merchant_counts = df['merchant'].value_counts().reset_index()
            merchant_counts.columns = ['merchant', 'count']
            results = merchant_counts.to_dict(orient='records')
        elif 'strftime' in query and 'GROUP BY date' in query:
            # Extract date from timestamp
            df['date'] = pd.to_datetime(
                df['timestamp']).dt.strftime('%Y-%m-%d')
            date_counts = df['date'].value_counts().reset_index()
            date_counts.columns = ['date', 'count']
            date_counts = date_counts.sort_values('date', ascending=False)
            results = date_counts.to_dict(orient='records')
        else:
            # Default to returning all fraud transactions
            results = df.to_dict(orient='records')

        return jsonify({
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
