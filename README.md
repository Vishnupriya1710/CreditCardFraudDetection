# CreditCardFraudDetection
# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using various machine learning models. The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building](#model-building)
    - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
    - [XGBoost](#xgboost)
    - [Random Forest](#random-forest)
6. [Results and Evaluation](#results-and-evaluation)
7. [Requirements](#requirements)
8. [Usage](#usage)

## Project Overview

Credit card fraud is a significant problem in the financial industry, and detecting fraudulent transactions is critical to protect customers and financial institutions. This project builds and evaluates several machine learning models to identify fraudulent transactions from a dataset of credit card transactions.

## Dataset Information

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It has 284,807 transactions with 30 features. The dataset is highly imbalanced, with only 492 frauds out of 284,807 transactions. 

The features include:

- `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: Result of a PCA transformation.
- `Amount`: Transaction amount.
- `Class`: 1 for fraudulent transactions, 0 otherwise.

## Exploratory Data Analysis (EDA)

During EDA, we performed the following steps:

1. Loaded the data and displayed the first few rows.
2. Checked the data types and null values.
3. Described the dataset to get statistical insights.
4. Visualized the distribution of the `Class` feature to understand the imbalance.
5. Analyzed the amount involved in normal and fraudulent transactions.
6. Visualized the distribution of transactions over time for both classes.
7. Created a heatmap to identify any high correlations among the features.

## Data Preprocessing

Data preprocessing steps included:

1. Splitting the data into training, validation, and test sets.
2. Standardizing the features using `StandardScaler`.
3. Calculating class weights to handle the imbalance in the dataset.

## Model Building

### Artificial Neural Network (ANN)

An ANN model was built using TensorFlow and Keras with the following architecture:

- Three hidden layers with 256 neurons each, ReLU activation, batch normalization, and dropout.
- An output layer with a sigmoid activation function.
- The model was compiled with binary cross-entropy loss and metrics including false positives, false negatives, precision, and recall.
- The model was trained for 300 epochs with early stopping and class weights to handle the imbalance.

### XGBoost

An XGBoost classifier was trained with the default parameters. The evaluation metric used was AUC-PR.

### Random Forest

A Random Forest classifier was trained with 100 estimators.

## Results and Evaluation

The models were evaluated on the test set, and their performance was compared using the F1-score, accuracy, precision, recall, and confusion matrix.

## Requirements

To run this project, you need the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- xgboost

You can install these packages using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost
```

## Usage

1. Clone this repository.
2. Install the required libraries.
3. Download the dataset from Kaggle and place it in the `input` directory.
4. Run the notebook or script to execute the code.

## Acknowledgments

- The dataset was provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- The project is inspired by the need to combat credit card fraud and protect financial transactions.

---

Feel free to explore and modify the code to improve the models or try different approaches. Happy coding!
