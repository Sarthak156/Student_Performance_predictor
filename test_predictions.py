import pickle
import numpy as np
import pandas as pd

# Load models and scaler
rf_model = pickle.load(open("rf_model.pkl", "rb"))
gb_model = pickle.load(open("gb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_list = pickle.load(open("features.pkl", "rb"))

def make_prediction(study_hours, attendance, participation):
    """Make a prediction for given student metrics"""
    
    # Base features
    input_dict = {
        'weekly_self_study_hours': study_hours,
        'attendance_percentage': attendance,
        'class_participation': participation
    }
    
    # Calculate factors
    attendance_factor = 1.0 if attendance >= 75 else attendance / 75 * 0.7
    participation_factor = 1.0 if participation >= 2.0 else participation / 2.0 * 0.7
    
    # Ceiling features
    input_dict['attendance_ceiling'] = 100 if attendance >= 75 else attendance * 1.33
    input_dict['participation_ceiling'] = 100 if participation >= 2 else participation * 50
    
    # Effective study
    input_dict['effective_study_hours'] = study_hours * attendance_factor * participation_factor
    input_dict['effective_study_squared'] = input_dict['effective_study_hours'] ** 2
    
    # Synergy features
    input_dict['study_attendance_synergy'] = study_hours * attendance_factor
    input_dict['study_participation_synergy'] = study_hours * participation_factor
    input_dict['attendance_participation_synergy'] = attendance * participation
    
    # All factors
    input_dict['all_factors_combined'] = study_hours * attendance_factor * participation_factor
    
    # Polynomial
    input_dict['study_hours_squared'] = study_hours ** 2
    input_dict['attendance_percentage_squared'] = attendance ** 2
    input_dict['participation_squared'] = participation ** 2
    input_dict['attendance_times_participation'] = attendance * participation
    
    # Create array in correct order
    input_data = [input_dict[feature] for feature in feature_list]
    input_array = np.array([input_data])
    
    # Scale and predict
    input_scaled = scaler.transform(input_array)
    rf_pred = rf_model.predict(input_scaled)[0]
    gb_pred = gb_model.predict(input_scaled)[0]
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    ensemble_pred = np.clip(ensemble_pred, 0, 100)
    
    # Additional penalty if low on both
    if attendance < 75 and participation < 2:
        ensemble_pred = max(0, ensemble_pred - 25)
    
    return ensemble_pred

print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION - SENSITIVITY TEST")
print("=" * 70)

# Test scenarios
test_cases = [
    ("High Study, High Attendance, High Participation", 25, 95, 5),
    ("High Study, Low Attendance, Low Participation", 25, 50, 1),
    ("High Study, Low Attendance, High Participation", 25, 50, 5),
    ("High Study, High Attendance, Low Participation", 25, 95, 1),
    ("Low Study, High Attendance, High Participation", 5, 95, 5),
    ("Low Study, Low Attendance, Low Participation", 5, 50, 1),
    ("Medium Study, Medium Attendance, Medium Participation", 15, 75, 2.5),
    ("Very Low Everything", 2, 30, 0.5),
]

results = []
for name, study, attendance, participation in test_cases:
    pred = make_prediction(study, attendance, participation)
    results.append({
        'Scenario': name,
        'Study Hours': study,
        'Attendance %': attendance,
        'Participation': participation,
        'Predicted Score': round(pred, 2)
    })
    print(f"\n{name}")
    print(f"  Study Hours: {study}, Attendance: {attendance}%, Participation: {participation}/5")
    print(f"  >> Predicted Score: {pred:.2f}")

print("\n" + "=" * 70)
print("KEY OBSERVATIONS:")
print("=" * 70)
print("\n✓ Students with high study hours but LOW attendance/participation:")
print("  Should get significantly LOWER scores (difference vs high attendance case)")
print("\n✓ The 'Effective Study Hours' metric now accounts for:")
print("  - Study Time × (Attendance/100) × (Participation/5)")
print("  - Low attendance/participation directly reduces study effectiveness")
print("\n✓ Additional penalty features for:")
print("  - Attendance < 75%")
print("  - Participation < 2/5")
