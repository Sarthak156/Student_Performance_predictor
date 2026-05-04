# __define-ocg__
import pandas as pd
import numpy as np

df = pd.read_csv("student_performance.csv")

print("=" * 60)
print("STUDENT PERFORMANCE DATASET ANALYSIS")
print("=" * 60)

print("\nColumns:\n", df.columns.tolist())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("FEATURE CORRELATIONS WITH TARGET (total_score)")
print("=" * 60)

features = [
    'weekly_self_study_hours',
    'attendance_percentage',
    'class_participation'
]

for feature in features:
    correlation = df[feature].corr(df['total_score'])
    print(f"{feature}: {correlation:.4f}")

print("\n" + "=" * 60)
print("GRADE DISTRIBUTION")
print("=" * 60)
print(df['grade'].value_counts().sort_index())

print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)
print(df.head(10))