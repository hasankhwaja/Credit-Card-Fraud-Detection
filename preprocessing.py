import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_fraud_data(file_path):
    """
    Preprocess the fraud dataset and split into training and validation sets.

    Parameters:
        file_path (str): Path to the input CSV file.

    Returns:
        tuple: Processed training and validation sets (X_train, X_val, y_train, y_val).
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'trans_date_trans_time' to datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

    # Calculate age using 'dob' and transaction year
    df['transaction_year'] = df['trans_date_trans_time'].dt.year
    df['year_of_birth'] = pd.to_datetime(df['dob']).dt.year
    df['age'] = df['transaction_year'] - df['year_of_birth']
    df.drop(columns=['dob', 'transaction_year', 'year_of_birth'], inplace=True)

    # Drop irrelevant columns
    irrelevant_columns = ['Unnamed: 0', 'cc_num', 'trans_num', 'street']
    df_cleaned = df.drop(columns=irrelevant_columns)

    # Haversine function to calculate distance
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers.
        return c * r

    # Calculate distance and add to the dataset
    df_cleaned['distance'] = df_cleaned.apply(
        lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)

    # Create bins for latitude and longitude
    n_bins = 10
    df_cleaned['lat_bucket'] = pd.cut(df_cleaned['lat'], bins=n_bins, labels=False)
    df_cleaned['long_bucket'] = pd.cut(df_cleaned['long'], bins=n_bins, labels=False)
    df_cleaned['merch_lat_bucket'] = pd.cut(df_cleaned['merch_lat'], bins=n_bins, labels=False)
    df_cleaned['merch_long_bucket'] = pd.cut(df_cleaned['merch_long'], bins=n_bins, labels=False)

    # Encode categorical columns
    categorical_columns = ['merchant', 'category', 'gender', 'job']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
        label_encoders[col] = le

    # Drop columns that are no longer needed
    columns_to_drop = ['trans_date_trans_time', 'first', 'last', 'city', 'state', 'zip', 'lat', 'long', 'merch_lat',
                       'merch_long']
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

    # Separate features and target variable
    X = df_cleaned.drop(columns=['is_fraud'])
    y = df_cleaned['is_fraud']

    # Normalize numerical columns
    numerical_columns = ['amt', 'age', 'distance', 'lat_bucket', 'long_bucket', 'merch_lat_bucket', 'merch_long_bucket']
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Split into training and validation sets (fixed parameters)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, scaler