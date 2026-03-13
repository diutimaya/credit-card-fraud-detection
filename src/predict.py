# predict.py

import joblib
import numpy as np

# Load the trained model
model = joblib.load("../models/fraud_detection_model.pkl")

print("Credit Card Fraud Detection System")
print("-----------------------------------")
print("Enter transaction details\n")

# The dataset has 30 input features
features = []

for i in range(30):
    value = float(input(f"Enter value for feature {i+1}: "))
    features.append(value)

# Convert input into numpy array
features = np.array(features).reshape(1, -1)

# Make prediction
prediction = model.predict(features)

# Show result
if prediction[0] == 1:
    print("\n🚨 Fraudulent Transaction Detected!")
else:
    print("\n✅ Normal Transaction")