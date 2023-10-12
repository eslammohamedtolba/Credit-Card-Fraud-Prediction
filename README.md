# Credit-Card-Fraud-Prediction
This repository contains a Python script that builds a credit card fraud prediction model using logistic regression.
The model is trained on a dataset of credit card transactions and is capable of classifying transactions as either legitimate or fraudulent and named 'creditcard.csv'.
The model achieves an accuracy of 94% on the test data.

## Prerequisites
Before running the code, make sure you have the following libraries and tools installed:
-pandas
-matplotlib
-seaborn
-scikit-learn
-numpy
you can install them using pip: pip install pandas matplotlib seaborn scikit-learn numpy

## Overview
The code performs the following steps to build and evaluate the credit card fraud prediction model:

1- Data Loading and Exploration
-Load the credit card transaction dataset from a CSV file.
-Display the first and last rows of the dataset.
-Check the shape and statistical information of the dataset.
-Check for missing values and perform data cleaning.

2- Data Visualization
-Visualize the distribution of the 'Class' feature to understand the data balance.
-Plot a distribution of the 'Class' feature to visualize the number of legitimate and fraudulent transactions.

3- Data Balancing
-Due to data imbalance, split the data into legitimate and fraud datasets.
-Balance the data by randomly selecting 204 legitimate transactions.
-Concatenate the legitimate and fraud datasets to create a balanced dataset.

4- Feature Correlation
-Calculate and visualize the correlation between all features in the dataset using a heatmap.

5- Data Splitting
-Split the dataset into input data (X) and labels (Y).
-Further split the data into training and test datasets.

6- Model Training and Evaluation
-Create a Logistic Regression model (LRModel) and train it using the training data.
-Make predictions on both the training and test datasets.
-Calculate and print the accuracy of the model on both sets.

7- Predictive System
-Create a predictive system to classify a given set of credit card transaction features.
-Provide an example input data and make predictions to classify the transaction as either legitimate or fraudulent.

## Model Accuracy
The accuracy of the credit card fraud prediction model on the test data is 94%.

## Contributions
Contributions to this repository are welcome. 
If you have any improvements or additional features to add to the code, feel free to submit a pull request.
