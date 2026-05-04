# Student Performance Predictor

## Project Description
Student Performance Predictor is a Flask-based machine learning web application that estimates a student's expected total score using study hours, attendance, and class participation. The project combines feature engineering with an ensemble of regression models to produce more realistic predictions and highlight how academic habits influence performance.

## Resume Summary
Built a full-stack machine learning application that predicts student performance from academic behavior data. Trained and deployed an ensemble regression model with engineered interaction features, then exposed it through a Flask API and browser interface for real-time score prediction.

## Key Features
- Predicts student score from three inputs: weekly self-study hours, attendance percentage, and class participation.
- Uses feature engineering to capture non-linear relationships and interaction effects.
- Combines Random Forest and Extra Trees models for more stable predictions.
- Provides a lightweight Flask web interface for easy interaction.
- Returns individual model predictions and an ensemble prediction through a JSON API.

## Project Structure
- `app.py` - Flask app that serves the UI and prediction endpoint.
- `Train_model.py` - training script that builds the models, scaler, and feature list.
- `test_predictions.py` - quick prediction sensitivity checks across sample scenarios.
- `explore.py` - data exploration and analysis script.
- `student_performance.csv` - dataset used for training and analysis.
- `templates/index.html` - frontend page for entering student details and viewing predictions.
- `rf_model.pkl`, `gb_model.pkl`, `scaler.pkl`, `features.pkl` - saved model artifacts used by the app.

## How It Works
1. The user enters study hours, attendance, and participation on the web form.
2. The Flask backend converts the inputs into the same engineered feature set used during training.
3. The inputs are scaled and passed through two trained regressors.
4. The app averages the predictions and applies guardrails for low attendance and participation.
5. The final predicted score is returned to the browser as JSON and displayed to the user.

## Tech Stack
- Python
- Flask
- NumPy
- Pandas
- scikit-learn
- HTML, CSS, JavaScript

## Setup and Installation
1. Create and activate a virtual environment.
2. Install the required packages:

```bash
pip install flask numpy pandas scikit-learn
```

3. Train the model artifacts if needed:

```bash
python Train_model.py
```

4. Start the Flask application:

```bash
python app.py
```

5. Open the app in your browser at:

```text
http://localhost:5000
```

## Example Input
- Weekly self-study hours: `15`
- Attendance percentage: `85`
- Class participation: `3`

## Example Output
- Predicted score with a grade-style interpretation in the UI.
- API response includes the ensemble prediction plus individual model outputs.

## Notes
- The training script saves model artifacts to `.pkl` files in the project root.
- The prediction pipeline in `app.py` must stay consistent with the feature engineering used in `Train_model.py`.
- `test_predictions.py` can be used to verify that the model reacts sensibly to different student behavior scenarios.

## Possible Improvements
- Add user authentication and prediction history.
- Show confidence bands or explanation features for each prediction.
- Replace the static frontend with a more polished dashboard.
- Add automated tests for the Flask prediction endpoint.
