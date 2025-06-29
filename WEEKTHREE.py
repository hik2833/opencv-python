"""
polynomial_regression_salary_predictor.py

Assignment: Build a Polynomial Regression Model to Predict Employee Salary

Instructions:
-------------
1. Make sure you have installed required packages:
   pip install pandas numpy matplotlib scikit-learn

2. Place your CSV file in the same folder. Name it: salary_data.csv
   It should have two columns (example):
       YearsExperience,Salary

3. Run this script:
   python polynomial_regression_salary_predictor.py

This script will:
 - Load and visualize the data
 - Fit both Linear and Polynomial Regression models
 - Compare their predictions visually
 - Predict salary for a new candidate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# === STEP 1: LOAD THE DATA ===
try:
    data = pd.read_csv('salary_data.csv')
except FileNotFoundError:
    print("\nERROR: Could not find 'salary_data.csv'. Make sure it's in the same folder.\n")
    exit()

# CHECK COLUMN NAMES
print("\nData columns:", data.columns.tolist())

# EDIT THESE IF YOUR CSV HAS DIFFERENT HEADERS:
X_column = 'YearsExperience'
y_column = 'Salary'

X = data[[X_column]].values
y = data[y_column].values

print("\nFirst 5 rows of data:")
print(data.head())

# === STEP 2: VISUALIZE THE RAW DATA ===
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.title('Salary vs. Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# === STEP 3: FIT LINEAR REGRESSION ===
lin_model = LinearRegression()
lin_model.fit(X, y)

print("\nLinear Regression model trained.")

# === STEP 4: FIT POLYNOMIAL REGRESSION ===
degree = 4  # Degree of the polynomial
poly_transformer = PolynomialFeatures(degree=degree)
X_poly = poly_transformer.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

print(f"\nPolynomial Regression model trained with degree = {degree}.")

# === STEP 5: VISUALIZE COMPARISON ===
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, lin_model.predict(X), color='green', label='Linear Regression')

# For smooth polynomial curve
X_grid = np.linspace(min(X), max(X), 200).reshape(-1, 1)
plt.plot(X_grid, poly_model.predict(poly_transformer.transform(X_grid)), color='red', label=f'Polynomial Regression (degree {degree})')

plt.title('Comparison of Linear and Polynomial Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# === STEP 6: PREDICT SALARY FOR A NEW CANDIDATE ===
# Example: predict for 6.5 years of experience
test_experience = 6.5

lin_pred = lin_model.predict([[test_experience]])[0]
poly_pred = poly_model.predict(poly_transformer.transform([[test_experience]]))[0]

print(f"\nPredicted salary for {test_experience} years experience:")
print(f"  - Linear Regression: ${lin_pred:,.2f}")
print(f"  - Polynomial Regression (degree {degree}): ${poly_pred:,.2f}")

# === STEP 7: OPTIONAL - USER INPUT FOR PREDICTION ===
while True:
    try:
        user_input = input("\nEnter years of experience to predict salary (or 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting program. Goodbye!")
            break
        years = float(user_input)
        lin_salary = lin_model.predict([[years]])[0]
        poly_salary = poly_model.predict(poly_transformer.transform([[years]]))[0]
        print(f"\nPredicted salary for {years} years:")
        print(f"  - Linear Regression: ${lin_salary:,.2f}")
        print(f"  - Polynomial Regression: ${poly_salary:,.2f}")
    except ValueError:
        print("Invalid input. Please enter a number or 'q' to quit.")
