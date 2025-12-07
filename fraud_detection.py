"""
Hands-On Assignment 4
Fraud Detection using Unsupervised Deep Learning (AutoEncoder)
Library: PyOD
"""

import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("creditcard.csv")

# Remove label column for unsupervised learning
X = data.drop(columns=["Class"])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize AutoEncoder (PyOD default configuration)
autoencoder = AutoEncoder()

# Train model
autoencoder.fit(X_scaled)

# Predict anomalies
predictions = autoencoder.predict(X_scaled)
scores = autoencoder.decision_function(X_scaled)

# Count detected frauds
fraud_detected = np.sum(predictions == 1)

print("Fraud Detection Results")
print("----------------------")
print(f"Total transactions: {len(data)}")
print(f"Detected fraudulent transactions: {fraud_detected}")

print("\nSample anomaly scores:")
print(scores[:10])
