# Silverwing ML
# Phase 2 - Lesson 21
# First Machine-Learning Model
# Linear Regression


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 21")
print("First Machine-Learning Model")
print()


# ==================================================
# 1. CREATE TRAINING DATA
# ==================================================

data = {
    "temperature": [
        65, 68, 72, 75, 78,
        80, 82, 85, 88, 90,
        92, 95, 98, 100, 102,
        105, 108, 110, 112, 115
    ],

    "pressure": [
        100, 102, 105, 108, 110,
        112, 115, 118, 120, 122,
        124, 126, 130, 132, 135,
        138, 140, 142, 145, 148
    ],

    "rpm": [
        1400, 1450, 1500, 1550, 1600,
        1650, 1700, 1800, 1900, 2000,
        2100, 2200, 2300, 2400, 2500,
        2700, 2800, 2900, 3000, 3200
    ],

    "operating_hours": [
        500, 700, 900, 1100, 1300,
        1500, 1700, 1900, 2100, 2300,
        2500, 2700, 2900, 3100, 3300,
        3500, 3700, 3900, 4100, 4500
    ],

    "risk_score": [
        0, 0, 5, 5, 10,
        10, 15, 20, 20, 25,
        25, 30, 35, 40, 50,
        60, 65, 70, 80, 100
    ]
}


df = pd.DataFrame(data)


# ==================================================
# 2. DISPLAY DATA
# ==================================================

print("TEST 1: Dataset")
print()

print(df)

print()


# ==================================================
# 3. SELECT FEATURES AND TARGET
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]


X = df[feature_columns]

y = df["risk_score"]


print("TEST 2: Features")
print()

print(X)

print()

print("Target:")
print(y)

print()


# ==================================================
# 4. SPLIT TRAINING AND TESTING DATA
# ==================================================

print("TEST 3: Train/Test Split")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print()


# ==================================================
# 5. CREATE THE MODEL
# ==================================================

print("TEST 4: Create Model")
print()


model = LinearRegression()


print(
    "Linear Regression model created."
)

print()


# ==================================================
# 6. TRAIN THE MODEL
# ==================================================

print("TEST 5: Training Model")
print()


model.fit(
    X_train,
    y_train
)


print(
    "Model training complete."
)

print()


# ==================================================
# 7. VIEW LEARNED PARAMETERS
# ==================================================

print("TEST 6: Learned Parameters")
print()


print(
    "Model coefficients:"
)

for feature, coefficient in zip(
        feature_columns,
        model.coef_
):

    print(
        feature,
        "->",
        coefficient
    )


print()

print(
    "Model intercept:",
    model.intercept_
)

print()


# ==================================================
# 8. MAKE PREDICTIONS
# ==================================================

print("TEST 7: Predictions")
print()


predictions = model.predict(
    X_test
)


results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions
})


print(results)

print()


# ==================================================
# 9. ROUND PREDICTIONS
# ==================================================

print("TEST 8: Rounded Predictions")
print()


results["predicted_rounded"] = (
    results["predicted"].round(2)
)


print(results)

print()


# ==================================================
# 10. MAE
# ==================================================

print("TEST 9: Mean Absolute Error")
print()


mae = mean_absolute_error(
    y_test,
    predictions
)


print(
    "MAE:",
    mae
)

print()


# ==================================================
# 11. MSE
# ==================================================

print("TEST 10: Mean Squared Error")
print()


mse = mean_squared_error(
    y_test,
    predictions
)


print(
    "MSE:",
    mse
)

print()


# ==================================================
# 12. RMSE
# ==================================================

print("TEST 11: Root Mean Squared Error")
print()


rmse = np.sqrt(mse)


print(
    "RMSE:",
    rmse
)

print()


# ==================================================
# 13. R-SQUARED
# ==================================================

print("TEST 12: R-Squared")
print()


r2 = r2_score(
    y_test,
    predictions
)


print(
    "R²:",
    r2
)

print()


# ==================================================
# 14. PREDICT A NEW MACHINE
# ==================================================

print("TEST 13: New Machine Prediction")
print()


new_machine = pd.DataFrame({
    "temperature": [97],
    "pressure": [128],
    "rpm": [2500],
    "operating_hours": [3000]
})


new_prediction = model.predict(
    new_machine
)


print(
    "New machine:"
)

print(new_machine)

print()

print(
    "Predicted risk score:",
    new_prediction[0]
)

print()


# ==================================================
# 15. CLASSIFY THE PREDICTION
# ==================================================

predicted_risk = new_prediction[0]


if predicted_risk >= 70:

    risk_level = "CRITICAL"

elif predicted_risk >= 40:

    risk_level = "HIGH"

elif predicted_risk >= 20:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


print(
    "Predicted risk level:",
    risk_level
)

print()


# ==================================================
# 16. COMPARE ACTUAL VS PREDICTED
# ==================================================

print("TEST 14: Prediction Comparison")
print()


for actual, predicted in zip(
        y_test,
        predictions
):

    error = actual - predicted

    print(
        "Actual:",
        round(actual, 2),
        "| Predicted:",
        round(predicted, 2),
        "| Error:",
        round(error, 2)
    )


print()


# ==================================================
# 17. SAVE THE RESULTS
# ==================================================

print("TEST 15: Save Predictions")
print()


results.to_csv(
    "model_predictions.csv",
    index=False
)


print(
    "Predictions saved as model_predictions.csv"
)

print()


# ==================================================
# 18. EXPLAIN WHAT THE MODEL LEARNED
# ==================================================

print("WHAT THE MODEL LEARNED")
print()

print(
    "The model learned numerical relationships "
    "between the input features and risk score."
)

print()

print(
    "Features:"
)

for feature in feature_columns:

    print(
        "-",
        feature
    )


print()

print(
    "Target:"
)

print(
    "- risk_score"
)

print()


# ==================================================
# 19. MACHINE LEARNING PIPELINE
# ==================================================

print("MACHINE LEARNING PIPELINE")
print()

print("Historical data")
print("      ↓")
print("Feature selection")
print("      ↓")
print("Train/test split")
print("      ↓")
print("Linear Regression")
print("      ↓")
print("Training")
print("      ↓")
print("Prediction")
print("      ↓")
print("Evaluation")
print("      ↓")
print("New machine prediction")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 21 COMPLETE ===")
