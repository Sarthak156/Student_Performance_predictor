# __define-ocg__
from flask import Flask, request, jsonify, render_template

import numpy as np
import os

app = Flask(__name__)

from Train_model import get_trained_objects

rf_model, gb_model, scaler, feature_list = get_trained_objects()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Base features from user input
        base_features = {
            'weekly_self_study_hours': float(data.get('weekly_self_study_hours', 0)),
            'attendance_percentage': float(data.get('attendance_percentage', 0)),
            'class_participation': float(data.get('class_participation', 0))
        }

        # Build input dictionary with engineered features
        input_dict = base_features.copy()
        
        # Calculate factors - LOW if attendance/participation are insufficient
        attendance_factor = (1.0 if input_dict['attendance_percentage'] >= 75 
                           else input_dict['attendance_percentage'] / 75 * 0.7)
        participation_factor = (1.0 if input_dict['class_participation'] >= 2.0 
                              else input_dict['class_participation'] / 2.0 * 0.7)
        
        # Ceiling features for visualization of requirements
        input_dict['attendance_ceiling'] = (100 if input_dict['attendance_percentage'] >= 75
                                           else input_dict['attendance_percentage'] * 1.33)
        input_dict['participation_ceiling'] = (100 if input_dict['class_participation'] >= 2
                                              else input_dict['class_participation'] * 50)
        
        # CRITICAL: Study effectiveness is capped at 70% if requirements not met
        input_dict['effective_study_hours'] = (
            input_dict['weekly_self_study_hours'] * attendance_factor * participation_factor
        )
        input_dict['effective_study_squared'] = input_dict['effective_study_hours'] ** 2
        
        # Synergy features
        input_dict['study_attendance_synergy'] = (
            input_dict['weekly_self_study_hours'] * attendance_factor
        )
        input_dict['study_participation_synergy'] = (
            input_dict['weekly_self_study_hours'] * participation_factor
        )
        input_dict['attendance_participation_synergy'] = (
            input_dict['attendance_percentage'] * input_dict['class_participation']
        )
        
        # All three factors
        input_dict['all_factors_combined'] = (
            input_dict['weekly_self_study_hours'] * attendance_factor * participation_factor
        )
        
        # Polynomial features
        input_dict['study_hours_squared'] = input_dict['weekly_self_study_hours'] ** 2
        input_dict['attendance_percentage_squared'] = input_dict['attendance_percentage'] ** 2
        input_dict['participation_squared'] = input_dict['class_participation'] ** 2
        input_dict['attendance_times_participation'] = (
            input_dict['attendance_percentage'] * input_dict['class_participation']
        )

        # Create input array in correct feature order
        input_data = [input_dict[feature] for feature in feature_list]
        input_array = np.array([input_data])
        
        # Scale
        input_scaled = scaler.transform(input_array)

        # Get predictions from both models
        rf_pred = rf_model.predict(input_scaled)[0]
        gb_pred = gb_model.predict(input_scaled)[0]
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Cap predictions to 0-100 range
        ensemble_pred = np.clip(ensemble_pred, 0, 100)
        
        # Additional penalty if low on both attendance AND participation
        if input_dict['attendance_percentage'] < 75 and input_dict['class_participation'] < 2:
            ensemble_pred = max(0, ensemble_pred - 25)

        return jsonify({
            "prediction": round(ensemble_pred, 2),
            "rf_prediction": round(np.clip(rf_pred, 0, 100), 2),
            "gb_prediction": round(np.clip(gb_pred, 0, 100), 2),
            "confidence": "High" if 0 <= ensemble_pred <= 100 else "Check input values",
            "factors_applied": {
                "attendance_factor": round(attendance_factor, 2),
                "participation_factor": round(participation_factor, 2),
                "effective_study_hours": round(input_dict['effective_study_hours'], 2),
                "note": "Study effectiveness capped at 70% if attendance < 75% OR participation < 2"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)