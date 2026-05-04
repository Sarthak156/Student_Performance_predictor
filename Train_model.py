# __define-ocg__
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset - use sample for faster training
url = "https://drive.google.com/uc?id=1IjnmDMY_Tz5kzQD9m5mDJ9Bvf_YpOUC4"
df = pd.read_csv(url)

# For large datasets, sample for faster training while maintaining quality
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)
    print(f"Using sample of 50,000 rows from {len(df)} total rows for faster training")
else:
    print(f"Using all {len(df)} rows")

# Base features
base_features = [
    'weekly_self_study_hours',
    'attendance_percentage',
    'class_participation'
]

target = 'total_score'

X = df[base_features].copy()
y = df[target]

# AGGRESSIVE: Apply penalty directly to score based on attendance/participation minimums
# This ensures the model learns that low attendance/participation leads to lower scores
y_adjusted = y.copy()

# Apply sliding penalty for attendance < 75%
low_attendance_mask = X['attendance_percentage'] < 75
y_adjusted[low_attendance_mask] = y[low_attendance_mask] - ((75 - X.loc[low_attendance_mask, 'attendance_percentage']) / 75 * 15)

# Apply sliding penalty for participation < 2
low_participation_mask = X['class_participation'] < 2
y_adjusted[low_participation_mask] = y[low_participation_mask] - ((2 - X.loc[low_participation_mask, 'class_participation']) / 2 * 15)

# Apply combined penalty for both low
both_low_mask = low_attendance_mask & low_participation_mask
y_adjusted[both_low_mask] = y[both_low_mask] - 25  # Heavy penalty

# Ensure adjusted scores stay in reasonable range
y_adjusted = np.clip(y_adjusted, 0, 100)

# Feature engineering
X['attendance_ceiling'] = np.where(X['attendance_percentage'] >= 75, 100, 
                                  X['attendance_percentage'] * 1.33)  # Scale up attendance contribution
X['participation_ceiling'] = np.where(X['class_participation'] >= 2, 100,
                                      X['class_participation'] * 50)  # Scale up participation contribution

# Effective study (multiplied by scaled attendance/participation)
attendance_factor = np.where(X['attendance_percentage'] >= 75, 1.0, 
                            X['attendance_percentage'] / 75 * 0.7)  # Max 70% if attendance < 75%
participation_factor = np.where(X['class_participation'] >= 2.0, 1.0,
                               X['class_participation'] / 2.0 * 0.7)  # Max 70% if participation < 2

X['effective_study_hours'] = X['weekly_self_study_hours'] * attendance_factor * participation_factor
X['effective_study_squared'] = X['effective_study_hours'] ** 2

# Aggressive synergy features that are zero when either factor is too low
X['study_attendance_synergy'] = X['weekly_self_study_hours'] * attendance_factor
X['study_participation_synergy'] = X['weekly_self_study_hours'] * participation_factor
X['attendance_participation_synergy'] = X['attendance_percentage'] * X['class_participation']

# All three factors multiplied - severely reduced if any factor is low
X['all_factors_combined'] = X['weekly_self_study_hours'] * attendance_factor * participation_factor

# Polynomial features
X['study_hours_squared'] = X['weekly_self_study_hours'] ** 2
X['attendance_percentage_squared'] = X['attendance_percentage'] ** 2
X['participation_squared'] = X['class_participation'] ** 2
X['attendance_times_participation'] = X['attendance_percentage'] * X['class_participation']

# Get all feature names after engineering
all_features = X.columns.tolist()

print(f"Total features: {len(all_features)}")
print(f"Features: {all_features}\n")

# Split - use y_adjusted instead of y for stronger penalty learning
X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Primary model: Random Forest (efficient for large datasets)
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

# Secondary model: Extra Trees (faster than GB, good ensemble component)
et_model = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

print("Training Extra Trees...")
et_model.fit(X_train_scaled, y_train)

# Predictions and evaluation
rf_pred = rf_model.predict(X_test_scaled)
et_pred = et_model.predict(X_test_scaled)

# Ensemble prediction (equal weight)
ensemble_pred = (rf_pred + et_pred) / 2

# Metrics
rf_r2 = r2_score(y_test, rf_pred)
et_r2 = r2_score(y_test, et_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
et_rmse = np.sqrt(mean_squared_error(y_test, et_pred))
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))

rf_mae = mean_absolute_error(y_test, rf_pred)
et_mae = mean_absolute_error(y_test, et_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"\nRandom Forest:")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  MAE: {rf_mae:.4f}")

print(f"\nExtra Trees:")
print(f"  R² Score: {et_r2:.4f}")
print(f"  RMSE: {et_rmse:.4f}")
print(f"  MAE: {et_mae:.4f}")

print(f"\nEnsemble (50% RF + 50% ET):")
print(f"  R² Score: {ensemble_r2:.4f}")
print(f"  RMSE: {ensemble_rmse:.4f}")
print(f"  MAE: {ensemble_mae:.4f}")

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 50)
feature_importance = sorted(zip(all_features, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
for feature, importance in feature_importance:
    print(f"  {feature}: {importance:.4f}")

print("\n" + "=" * 50)

# Save all models + scaler
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(et_model, open("gb_model.pkl", "wb"))  # Save ET as gb_model for compatibility
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(all_features, open("features.pkl", "wb"))

print("\nModels trained and saved successfully!")
print(f"Features saved for prediction consistency.")

def get_trained_objects():
    return rf_model, et_model, scaler, all_features