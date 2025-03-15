import csv
import json
import time
from kafka import KafkaProducer

# Initialize the Kafka producer with JSON serialization.
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

with open('fraudTest.csv', mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        # Convert fields to appropriate types.
        try:
            # Use the first column (often an index) as transaction_id.
            row["transaction_id"] = int(row["Unnamed: 0"])
            # Convert amount, timestamp, etc.
            row["amt"] = float(row["amt"])
            row["unix_time"] = int(row["unix_time"])
            # Convert latitude/longitude fields if available.
            row["lat"] = float(row["lat"]) if row["lat"] else None
            row["long"] = float(row["long"]) if row["long"] else None
            row["city_pop"] = int(row["city_pop"]) if row["city_pop"] else None
            row["merch_lat"] = float(row["merch_lat"]) if row["merch_lat"] else None
            row["merch_long"] = float(row["merch_long"]) if row["merch_long"] else None
            row["is_fraud"] = int(row["is_fraud"])
        except Exception as e:
            print("Error converting row:", row, e)
        
        print("Producing transaction:", row)
        producer.send("transactions", row)
        time.sleep(1)  # Adjust delay to simulate streaming

producer.close()
