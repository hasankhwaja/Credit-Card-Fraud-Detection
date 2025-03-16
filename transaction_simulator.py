import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import time


class TransactionSimulator:
    def __init__(self, db_path='transactions.db'):
        self.db_path = db_path
        self.categories = [
            'grocery_pos', 'shopping_pos', 'entertainment', 'food_dining',
            'travel', 'gas_transport', 'health_fitness', 'electronics'
        ]
        self.initialize_db()

    def initialize_db(self):
        """Create the transactions table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS transactions
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             transaction_date TEXT,
             category TEXT,
             amount REAL,
             merchant_lat REAL,
             merchant_long REAL,
             age INTEGER,
             city_pop INTEGER,
             merchant TEXT,
             job TEXT,
             processed INTEGER DEFAULT 0)
        ''')
        conn.commit()
        conn.close()

    def generate_transaction(self):
        """Generate a single realistic transaction"""
        now = datetime.now()

        # Generate realistic transaction data
        transaction = {
            'transaction_date': now.strftime('%Y-%m-%d %H:%M:%S'),
            'category': np.random.choice(self.categories),
            # Most transactions small, some large
            'amount': round(np.random.exponential(50) + 10, 2),
            'merchant_lat': np.random.uniform(25, 50),  # US latitude range
            # US longitude range
            'merchant_long': np.random.uniform(-130, -70),
            'age': np.random.randint(18, 85),
            'city_pop': np.random.randint(1000, 9000000),
            'merchant': f"Merchant_{np.random.randint(1, 100)}",
            'job': f"Job_{np.random.randint(1, 20)}"
        }

        return transaction

    def generate_and_save_transactions(self, num_transactions=1):
        """Generate and save multiple transactions to the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for _ in range(num_transactions):
            transaction = self.generate_transaction()
            c.execute('''
                INSERT INTO transactions
                (transaction_date, category, amount, merchant_lat, merchant_long,
                 age, city_pop, merchant, job)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction['transaction_date'],
                transaction['category'],
                transaction['amount'],
                transaction['merchant_lat'],
                transaction['merchant_long'],
                transaction['age'],
                transaction['city_pop'],
                transaction['merchant'],
                transaction['job']
            ))

        conn.commit()
        conn.close()

    def get_unprocessed_transactions(self, limit=20):
        """Retrieve unprocessed transactions from the database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM transactions 
            WHERE processed = 0 
            ORDER BY transaction_date 
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df

    def mark_transactions_as_processed(self, transaction_ids):
        """Mark transactions as processed"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            UPDATE transactions 
            SET processed = 1 
            WHERE id IN ({})
        '''.format(','.join(['?'] * len(transaction_ids))), transaction_ids)
        conn.commit()
        conn.close()

# Create background transaction generator


def generate_background_transactions(simulator, interval_seconds=5):
    """Generate 1-3 transactions every interval_seconds"""
    while True:
        num_transactions = np.random.randint(1, 4)
        simulator.generate_and_save_transactions(num_transactions)
        time.sleep(interval_seconds)
