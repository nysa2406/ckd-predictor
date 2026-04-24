# =========================
# CKD MODEL TRAINING (FINAL)
# =========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv('kidney_disease.csv', sep='\t')

# Replace missing values
# Replace missing values
df.replace('?', np.nan, inplace=True)

# Convert numeric columns properly
numeric_cols = ['sg', 'hemo', 'pcv', 'sc', 'al', 'bgr', 'rc', 'sod']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode categorical columns
# Encode categorical columns
df['dm'] = df['dm'].map({'yes':1, 'no':0})
df['htn'] = df['htn'].map({'yes':1, 'no':0})

df['classification'] = df['classification'].str.strip()
df['classification'] = df['classification'].map({'ckd':1, 'notckd':0})

# Remove rows where target is missing
df = df.dropna(subset=['classification'])

# Select ONLY features used in app
features = ['sg', 'hemo', 'pcv', 'sc', 'al', 'dm', 'bgr', 'rc', 'htn', 'sod']

df = df[features + ['classification']]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[features] = imputer.fit_transform(df[features])

# Split
X = df[features]
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + scaler
joblib.dump(model, 'ckd_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model trained and saved!")
