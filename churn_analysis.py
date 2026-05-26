# ============================================================
#  CUSTOMER CHURN PREDICTOR - Full ML Project
#  Portfolio Project | Data Analytics + Machine Learning
# ============================================================
#
#  HOW TO RUN:
#  1. Make sure churn.csv is inside your data/ folder
#  2. Open VS Code terminal
#  3. Run this command:
#     C:/Users/nitis/AppData/Local/Python/pythoncore-3.14-64/python.exe churn_analysis.py
#
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

print("=" * 55)
print("   CUSTOMER CHURN PREDICTOR")
print("=" * 55)

# ============================================================
#  STEP 1: LOAD DATA
# ============================================================

df = pd.read_csv('data/churn.csv')
print(f"\nData loaded: {df.shape[0]} customers, {df.shape[1]} features")
print(f"Columns: {df.columns.tolist()}")

# ============================================================
#  STEP 2: CLEAN DATA
# ============================================================

# Convert TotalCharges to number (it has some spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values with median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Fill any remaining missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop customerID - not useful for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert Churn to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")
print(f"Churn rate: {df['Churn'].mean()*100:.1f}% of customers churned")

# ============================================================
#  STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n--- Customer Overview ---")
print(f"Total customers    : {len(df):,}")
print(f"Churned customers  : {df['Churn'].sum():,}")
print(f"Retained customers : {(df['Churn']==0).sum():,}")
print(f"Avg monthly charge : ${df['MonthlyCharges'].mean():.2f}")
print(f"Avg tenure         : {df['tenure'].mean():.1f} months")

print("\n--- Churn by Contract Type ---")
print(df.groupby('Contract')['Churn'].mean().mul(100).round(1).astype(str) + '%')

print("\n--- Churn by Internet Service ---")
print(df.groupby('InternetService')['Churn'].mean().mul(100).round(1).astype(str) + '%')

# ============================================================
#  STEP 4: PREPARE DATA FOR ML
# ============================================================

# Encode all text columns to numbers
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing samples  : {len(X_test):,}")

# ============================================================
#  STEP 5: TRAIN MODELS
# ============================================================

print("\n--- Training Models ---")

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
print(f"Logistic Regression  → Accuracy: {lr_acc*100:.1f}%  AUC: {lr_auc:.3f}")

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print(f"Random Forest        → Accuracy: {rf_acc*100:.1f}%  AUC: {rf_auc:.3f}")

# Best model
best_model = rf if rf_acc > lr_acc else lr
best_pred = rf_pred if rf_acc > lr_acc else lr_pred
best_name = "Random Forest" if rf_acc > lr_acc else "Logistic Regression"
print(f"\nBest model: {best_name}")

# ============================================================
#  STEP 6: MODEL EVALUATION
# ============================================================

print("\n--- Classification Report ---")
print(classification_report(y_test, best_pred,
      target_names=['Stayed', 'Churned']))

# ============================================================
#  STEP 7: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Chart 1: Churn Distribution
ax = axes[0, 0]
churn_counts = df['Churn'].value_counts()
colors = ['#2563EB', '#DC2626']
ax.pie(churn_counts, labels=['Stayed', 'Churned'],
       autopct='%1.1f%%', colors=colors,
       wedgeprops=dict(width=0.6, edgecolor='white'))
ax.set_title('Customer Churn Distribution', fontweight='bold')

# Chart 2: Confusion Matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Stayed', 'Churned'],
            yticklabels=['Stayed', 'Churned'])
ax.set_title(f'Confusion Matrix ({best_name})', fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# Chart 3: Feature Importance (Random Forest)
ax = axes[1, 0]
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top10 = feat_imp.sort_values(ascending=True).tail(10)
top10.plot(kind='barh', ax=ax, color='#2563EB', alpha=0.85)
ax.set_title('Top 10 Features for Churn Prediction', fontweight='bold')
ax.set_xlabel('Importance Score')

# Chart 4: Monthly Charges vs Churn
ax = axes[1, 1]
# Use original numeric values
stayed = df[df['Churn']==0]['MonthlyCharges']
churned = df[df['Churn']==1]['MonthlyCharges']
ax.hist(stayed, bins=30, alpha=0.6, color='#2563EB', label='Stayed')
ax.hist(churned, bins=30, alpha=0.6, color='#DC2626', label='Churned')
ax.set_title('Monthly Charges: Stayed vs Churned', fontweight='bold')
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Number of Customers')
ax.legend()

plt.tight_layout()
plt.savefig('churn_dashboard.png', dpi=150, bbox_inches='tight')
print("\nDashboard saved as: churn_dashboard.png")

# ============================================================
#  STEP 8: BUSINESS INSIGHTS
# ============================================================

print("\n" + "="*55)
print("  BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*55)

print(f"\n1. CHURN RATE: {df['Churn'].mean()*100:.1f}% of customers are leaving.")
print("   Industry average is ~20% — compare your result.")

print(f"\n2. MODEL PERFORMANCE: {best_name} achieved")
print(f"   {max(rf_acc, lr_acc)*100:.1f}% accuracy in predicting churn.")

# Top 3 features
top3 = feat_imp.sort_values(ascending=False).head(3)
print(f"\n3. TOP CHURN FACTORS:")
for i, (feat, score) in enumerate(top3.items(), 1):
    print(f"   {i}. {feat} (importance: {score:.3f})")

print("\n4. RECOMMENDATIONS:")
print("   → Target month-to-month contract customers first")
print("   → Offer discounts to high monthly charge customers")
print("   → Focus retention on customers with tenure < 12 months")
print("   → Customers with fiber optic internet churn more")
print("="*55)
