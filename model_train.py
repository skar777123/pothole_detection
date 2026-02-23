import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# --- 1. Generate Synthetic Data ---
# We simulate 'windows' of data. Each window is 20 readings.
# Class 0: Flat Road (Noise only)
# Class 1: Pothole (Dip in the middle)

def generate_synthetic_data(n_samples=1000, window_size=20):
    X = []
    y = []
    
    baseline_height = 100  # cm (Sensor mounting height)
    
    for _ in range(n_samples):
        # 50% chance of being a pothole
        if np.random.rand() > 0.5:
            # Create Pothole: Baseline -> Drop -> Rise -> Baseline
            segment = np.random.normal(baseline_height, 2, window_size) # Normal noise
            
            # Create the dip
            pothole_start = np.random.randint(5, 15)
            pothole_width = np.random.randint(3, 6)
            depth = np.random.randint(5, 15) # 5cm to 15cm deep
            
            segment[pothole_start : pothole_start + pothole_width] += depth 
            # Note: Distance INCREASES when there is a pothole (ground is further away)
            
            X.append(segment)
            y.append(1) # Label: Pothole
        else:
            # Flat Road
            segment = np.random.normal(baseline_height, 2, window_size)
            X.append(segment)
            y.append(0) # Label: Road

    return np.array(X), np.array(y)

# --- 2. Train Model ---
print("Generating synthetic data...")
X, y = generate_synthetic_data()

# Feature Extraction (Optional: raw data works well for RF, but stats help)
# We will feed raw window data into the RF for simplicity.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training Random Forest...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)

print("Model Evaluation:")
print(classification_report(y_test, clf.predict(X_test)))

# --- 3. Save Model ---
joblib.dump(clf, 'pothole_model.pkl')
print("Model saved as 'pothole_model.pkl'")