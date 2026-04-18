import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Data Loading
df = pd.read_csv("diabetes.csv")
print("Shape:", df.shape)
print(df.head())

# 2. Data Preprocessing
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

imputer = SimpleImputer(strategy="median")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# 4. Primary Model Selection
# Random Forest chosen because:
# it Works well with tabular data
# it Handles non-linearity
# it Robust to noise

# 5. Cross Validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print("CV Mean:", cv_scores.mean())
print("CV Std:", cv_scores.std())

# 6. Hyperparameter Tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)

# 7. Best Model Selection
best_model = grid.best_estimator_

# 8. Evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Save Model
with open("Diabetes_rf_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Model saved successfully!")
