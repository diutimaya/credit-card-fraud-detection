# Credit Card Fraud Detection

## Project Overview

This project develops a **machine learning model to detect fraudulent credit card transactions** using historical transaction data.

Credit card fraud detection is a critical challenge in financial systems due to the **highly imbalanced nature of transaction data**, where fraudulent transactions represent only a small fraction of total activity.

The objective of this project is to build and evaluate machine learning models capable of **accurately identifying fraudulent transactions while minimizing false positives**.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Handling imbalanced data using SMOTE
- Training multiple classification models
- Model evaluation and comparison
- Saving the best-performing model for prediction

---

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** containing transactions made by European cardholders.

Dataset characteristics:

- **284,807 transactions**
- **492 fraudulent transactions**

Features include:

- **Time** – Seconds elapsed between transactions
- **Amount** – Transaction amount
- **V1–V28** – PCA-transformed features for confidentiality
- **Class** – Target variable

```
0 → Normal Transaction  
1 → Fraudulent Transaction
```

Due to privacy considerations, original transaction attributes were transformed using **Principal Component Analysis (PCA)**.

### Dataset Access

The dataset is not included in this repository due to GitHub's file size limitations.

Download it from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place the dataset in the following directory:

```
data/creditcard.csv
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Jupyter Notebook
- Git & GitHub

---

## Project Structure

```
credit-card-fraud-detection
│
├── data
│   └── creditcard.csv
│
├── notebook
│   └── fraud_detection.ipynb
│
├── src
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models
│   └── fraud_detection_model.pkl
│
├── images
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_comparison.png
│
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

The following preprocessing steps were applied:

- Verified that the dataset contains **no missing values**
- Separated **features and target variable**
- Performed **train-test split**
- Applied **StandardScaler** for feature normalization
- Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**

Handling the imbalance significantly improves the model’s ability to detect fraudulent transactions.

---

## Exploratory Data Analysis

Exploratory analysis was conducted to understand transaction behavior and dataset characteristics.

Key analyses include:

- Fraud vs Normal transaction distribution
- Transaction amount distribution
- Feature correlation analysis

These insights help reveal patterns and confirm the **severe class imbalance present in the dataset**.

---

## Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Each model was evaluated using metrics appropriate for **imbalanced classification problems**.

---

## Model Evaluation

Model performance was evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

Among the evaluated models, **Random Forest demonstrated the best overall performance for fraud detection**.

---

## Key Visualizations

The project includes multiple visualizations to better understand the dataset and model performance:

- Fraud vs Normal Transaction Distribution
- Transaction Amount Distribution
- Feature Correlation Heatmap
- Confusion Matrix
- ROC Curve
- Model Performance Comparison

---

## Model Prediction

The trained model is saved using **Joblib** and can be used to predict whether a new transaction is fraudulent.

Example output:

```
Fraudulent Transaction
```

or

```
Normal Transaction
```

---

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/diutimaya/credit-card-fraud-detection.git
```

Navigate to the project directory:

```bash
cd credit-card-fraud-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

---

## Future Improvements

Potential extensions of this project include:

- Deploying the model as a real-time prediction API
- Using advanced models such as **XGBoost or LightGBM**
- Implementing anomaly detection techniques
- Building a real-time fraud monitoring dashboard

---

## Author

**Diutimaya Mohanty**  
B.Tech Student — Data Science Specialization
