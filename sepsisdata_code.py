import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# -------------------------------
# 1. Load dataset
# -------------------------------
file_path = r"C:\Users\Shrey\Downloads\cloned_sepsis_dataset.csv"
df = pd.read_csv(file_path)

# Standardize column names
df.columns = [c.lower().strip() for c in df.columns]

# -------------------------------
# 2. Features and labels
# -------------------------------
features = ["heart_rate", "respiratory_rate", "body_temperature", "oxygen_saturation"]
target = "sepsis_label"

X = df[features]
y = df[target]

# -------------------------------
# 3. Train (first 1500) / Validation (next 500)
# -------------------------------
X_train, y_train = X.iloc[:1500], y.iloc[:1500]
X_val, y_val = X.iloc[1500:2000], y.iloc[1500:2000]

# -------------------------------
# 4. Preprocessing
# -------------------------------
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)

# -------------------------------
# 5. Train RandomForest
# -------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_prep, y_train)

# -------------------------------
# 6. Evaluate
# -------------------------------
y_pred = model.predict(X_val_prep)
y_proba = model.predict_proba(X_val_prep)[:, 1]

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

print("\n✅ Model Performance on Validation Set (500 samples):")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"ROC AUC  : {auc:.3f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# -------------------------------
# 7. Save trained model
# -------------------------------
with open("sepsis_rf_model.pkl", "wb") as f:
    pickle.dump({"model": model, "preprocessor": preprocessor, "features": features}, f)

print("\n📂 Model saved as: sepsis_rf_model.pkl")
