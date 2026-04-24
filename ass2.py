# =========================
# CKD Progression Predictor
# =========================

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load Dataset
# =========================

df = pd.read_csv('kidney_disease.csv', sep='\t')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# =========================
# Data Cleaning
# =========================

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop ID column if exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Convert numeric columns properly
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# =========================
# Handle Missing Values
# =========================

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Numeric → mean
imputer_num = SimpleImputer(strategy='mean')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Categorical → most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# =========================
# Encode Categorical Data
# =========================

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# =========================
# Target Variable
# =========================

# Assuming 'classification' column exists
# ckd = 1, notckd = 0

if 'classification' in df.columns:
    y = df['classification']
    X = df.drop('classification', axis=1)
else:
    print("⚠️ Please check target column name!")

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Model Training
# =========================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================
# Prediction
# =========================

y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# Feature Importance
# =========================

importances = model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:")
print(feat_df.head(10))

# =========================
# Sample Prediction
# =========================

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Prediction:")
if prediction[0] == 1:
    print("⚠️ High Risk: Likely CKD Progression (Stage 4–5)")
else:
    print("✅ Low Risk: Stable CKD")

    import joblib

joblib.dump(model, 'ckd_model.pkl')
joblib.dump(scaler, 'scaler.pkl')