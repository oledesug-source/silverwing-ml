# Silverwing ML
# Phase 2 - Lesson 22
# Model Performance, Overfitting and Underfitting

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
print("Phase 2 - Lesson 22")
print("Model Performance and Generalization")
print()


# ==================================================
# 1. CREATE DATASET
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
# 2. FEATURES AND TARGET
# ==================================================

features = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]


X = df[features]

y = df["risk_score"]


print("FEATURES")
print(X)

print()

print("TARGET")
print(y)

print()


# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("TRAINING SAMPLES:", len(X_train))
print("TESTING SAMPLES:", len(X_test))
print()


# ==================================================
# 4. TRAIN MODEL
# ==================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


print("Model training complete.")
print()


# ==================================================
# 5. TRAINING PREDICTIONS
# ==================================================

train_predictions = model.predict(
    X_train
)


# ==================================================
# 6. TESTING PREDICTIONS
# ==================================================

test_predictions = model.predict(
    X_test
)


# ==================================================
# 7. TRAINING METRICS
# ==================================================

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_mse = mean_squared_error(
    y_train,
    train_predictions
)

train_rmse = np.sqrt(train_mse)

train_r2 = r2_score(
    y_train,
    train_predictions
)


# ==================================================
# 8. TESTING METRICS
# ==================================================

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ==================================================
# 9. DISPLAY TRAINING PERFORMANCE
# ==================================================

print("TRAINING PERFORMANCE")
print()

print("MAE:", train_mae)
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R²:", train_r2)

print()


# ==================================================
# 10. DISPLAY TESTING PERFORMANCE
# ==================================================

print("TESTING PERFORMANCE")
print()

print("MAE:", test_mae)
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R²:", test_r2)

print()


# ==================================================
# 11. COMPARE PERFORMANCE
# ==================================================

print("TRAIN VS TEST")
print()

print(
    "Training R²:",
    train_r2
)

print(
    "Testing R²:",
    test_r2
)

print(
    "Training RMSE:",
    train_rmse
)

print(
    "Testing RMSE:",
    test_rmse
)

print()


# ==================================================
# 12. SIMPLE GENERALIZATION CHECK
# ==================================================

print("GENERALIZATION CHECK")
print()

r2_difference = (
        train_r2
        -
        test_r2
)

rmse_difference = (
        test_rmse
        -
        train_rmse
)


print(
    "R² difference:",
    r2_difference
)

print(
    "RMSE difference:",
    rmse_difference
)

print()


# ==================================================
# 13. SIMPLE INTERPRETATION
# ==================================================

if r2_difference > 0.20:

    print(
        "Possible overfitting detected: "
        "training performance is substantially "
        "better than testing performance."
    )

elif test_r2 < 0:

    print(
        "Testing performance is poor. "
        "The model may not generalize well."
    )

else:

    print(
        "No strong overfitting signal detected "
        "from this simple check."
    )

print()


# ==================================================
# 14. ACTUAL VS PREDICTED
# ==================================================

print("ACTUAL VS PREDICTED")
print()

results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": test_predictions
})

results["error"] = (
        results["actual"]
        -
        results["predicted"]
)

results["absolute_error"] = (
    results["error"].abs()
)

print(
    results.round(2)
)

print()


# ==================================================
# 15. LARGEST PREDICTION ERROR
# ==================================================

largest_error_index = (
    results["absolute_error"].idxmax()
)

largest_error = results.loc[
    largest_error_index
]


print("LARGEST TEST ERROR")
print()

print(
    "Actual:",
    largest_error["actual"]
)

print(
    "Predicted:",
    round(
        largest_error["predicted"],
        2
    )
)

print(
    "Absolute error:",
    round(
        largest_error["absolute_error"],
        2
    )
)

print()


# ==================================================
# 16. WHAT IS OVERFITTING?
# ==================================================

print("OVERFITTING")
print()

print(
    "Overfitting occurs when a model learns "
    "the training data too closely and performs "
    "much worse on unseen data."
)

print()


# ==================================================
# 17. WHAT IS UNDERFITTING?
# ==================================================

print("UNDERFITTING")
print()

print(
    "Underfitting occurs when a model is too "
    "simple to capture important patterns."
)

print()


# ==================================================
# 18. WHAT IS GENERALIZATION?
# ==================================================

print("GENERALIZATION")
print()

print(
    "Generalization means performing well "
    "on new data that was not used for training."
)

print()


# ==================================================
# 19. MACHINE LEARNING WORKFLOW
# ==================================================

print("ML MODEL DEVELOPMENT")
print()

print("1. Collect representative data")
print("2. Clean the data")
print("3. Split training and testing data")
print("4. Train the model")
print("5. Evaluate on unseen data")
print("6. Check for overfitting")
print("7. Improve the model")
print("8. Validate again")

print()


# ==================================================
# 20. SAVE RESULTS
# ==================================================

results.to_csv(
    "lesson22_predictions.csv",
    index=False
)

print(
    "Prediction results saved as "
    "lesson22_predictions.csv"
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 22 COMPLETE ===")
