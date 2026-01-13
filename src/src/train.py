import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/plants.csv")


# =========================
# Feature Engineering
# =========================
df["sunlight_per_water"] = df["sunlight_hours"] / (df["watering_freq_per_week"] + 1)
df["watering_squared"] = df["watering_freq_per_week"] ** 2


# =========================
# Target Encoding
# =========================
target_encoder = LabelEncoder()
df["water_need"] = target_encoder.fit_transform(df["water_need"])


# =========================
# Split Features / Target
# =========================
X = df.drop("water_need", axis=1)
y = df["water_need"]


# =========================
# Stratified Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# =========================
# Handle Imbalance
# =========================
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)


# =========================
# Model
# =========================
model = XGBClassifier(
    n_estimators=700,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)


# =========================
# Evaluation
# =========================
pred = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, pred))
print("Macro F1:", f1_score(y_test, pred, average="macro"))


# =========================
# Cross Validation (Proof of ML maturity)
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

print("Cross-validated Macro F1:", scores.mean())
