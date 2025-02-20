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

## New Additions: Naive Bayes Implementation

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

By adding the Naive Bayes model, we aim to complement existing methods (e.g., Logistic Regression, Random Forest, and XGBoost) and compare how each performs under severe class imbalance.

---






