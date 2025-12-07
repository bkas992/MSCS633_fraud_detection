**Files Included**
  fraud_detection.py – Python code to train the AutoEncoder and detect fraud
  manifest.txt – Description of tools, libraries, and environment used

**Dataset**
  The anonymized credit card transactions dataset was obtained from Kaggle. The dataset file (creditcard.csv) is not included in this repository.

**How to Run**
1. Install required libraries (PyOD, numpy, pandas, scikit learn, torch)
2. Place creditcard.csv in the same directory as the Python file
3. Run the program using:
  python3 fraud_detection.py

The program trains an AutoEncoder model and prints the total number of transactions, detected fraudulent transactions, and sample anomaly scores.
