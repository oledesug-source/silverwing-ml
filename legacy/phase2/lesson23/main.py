# Silverwing ML
# Phase 2 - Lesson 23
# Classification
# Predicting Machine Risk Categories

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 23")
print("Classification")
print()


# ==================================================
# 1. CREATE MACHINE DATA
# ==================================================

data = {
    "temperature": [
        60, 62, 65, 68, 70,
        72, 74, 76, 78, 80,
        82, 84, 86, 88, 90,
        92, 94, 96, 98, 100,
        102, 104, 106, 108, 110
    ],

    "pressure": [
        95, 98, 100, 102, 105,
        108, 110, 112, 115, 118,
        120, 122, 124, 126, 128,
        130, 132, 134, 136, 138,
        140, 142, 144, 146, 148
    ],

    "rpm": [
        1200, 1250, 1300, 1350, 1400,
        1450, 1500, 1550, 1600, 1700,
        1800, 1900, 2000, 2100, 2200,
        2300, 2400, 2500, 2600, 2750,
        2900, 3000, 3100, 3200, 3400
    ],

    "operating_hours": [
        200, 400, 600, 800, 1000,
        1200, 1400, 1600, 1800, 2000,
        2200, 2400, 2600, 2800, 3000,
        3200, 3400, 3600, 3800, 4000,
        4200, 4400, 4600, 4800, 5000
    ],

    "risk_level": [
        "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL",
        "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL",
        "NORMAL", "NORMAL", "NORMAL", "WARNING", "WARNING",
        "WARNING", "WARNING", "WARNING", "WARNING", "WARNING",
        "CRITICAL", "CRITICAL", "CRITICAL", "CRITICAL", "CRITICAL"
    ]
}


df = pd.DataFrame(data)


# ==================================================
# 2. DISPLAY DATASET
# ==================================================

print("TEST 1: Dataset")
print()

print(df)

print()


# ==================================================
# 3. CHECK CLASS DISTRIBUTION
# ==================================================

print("TEST 2: Class Distribution")
print()

print(
    df["risk_level"].value_counts()
)

print()


# ==================================================
# 4. SELECT FEATURES
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]

X = df[
    feature_columns
]

y = df[
    "risk_level"
]


print("TEST 3: Features")
print()

print(X)

print()


print("Target:")
print(y)

print()


# ==================================================
# 5. TRAIN / TEST SPLIT
# ==================================================

print("TEST 4: Train/Test Split")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
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
# 6. CREATE CLASSIFICATION MODEL
# ==================================================

print("TEST 5: Create Classifier")
print()


model = LogisticRegression(
    max_iter=1000
)


print(
    "Logistic Regression classifier created."
)

print()


# ==================================================
# 7. TRAIN MODEL
# ==================================================

print("TEST 6: Training Classifier")
print()


model.fit(
    X_train,
    y_train
)


print(
    "Classification model training complete."
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
    "actual": y_test.to_numpy(),
    "predicted": predictions
})


print(results)

print()


# ==================================================
# 9. ACCURACY
# ==================================================

print("TEST 8: Accuracy")
print()


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    "Accuracy:",
    accuracy
)

print(
    "Accuracy percentage:",
    accuracy * 100,
    "%"
)

print()


# ==================================================
# 10. PRECISION
# ==================================================

print("TEST 9: Precision")
print()


precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    "Weighted precision:",
    precision
)

print()


# ==================================================
# 11. RECALL
# ==================================================

print("TEST 10: Recall")
print()


recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    "Weighted recall:",
    recall
)

print()


# ==================================================
# 12. F1 SCORE
# ==================================================

print("TEST 11: F1 Score")
print()


f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print(
    "Weighted F1:",
    f1
)

print()


# ==================================================
# 13. CLASSIFICATION REPORT
# ==================================================

print("TEST 12: Classification Report")
print()

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print()


# ==================================================
# 14. CONFUSION MATRIX
# ==================================================

print("TEST 13: Confusion Matrix")
print()


labels = [
    "NORMAL",
    "WARNING",
    "CRITICAL"
]


matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


print(
    "Labels:",
    labels
)

print()

print(matrix)

print()


# ==================================================
# 15. CLASS PROBABILITIES
# ==================================================

print("TEST 14: Class Probabilities")
print()


probabilities = model.predict_proba(
    X_test
)


probability_columns = [
    f"probability_{label}"
    for label in model.classes_
]


probability_df = pd.DataFrame(
    probabilities,
    columns=probability_columns
)


print(probability_df.round(3))

print()


# ==================================================
# 16. PREDICT A NEW MACHINE
# ==================================================

print("TEST 15: New Machine")
print()


new_machine = pd.DataFrame({
    "temperature": [97],
    "pressure": [130],
    "rpm": [2600],
    "operating_hours": [3500]
})


new_prediction = model.predict(
    new_machine
)


new_probabilities = model.predict_proba(
    new_machine
)


print("New machine:")
print(new_machine)

print()

print(
    "Predicted risk level:",
    new_prediction[0]
)

print()

print(
    "Class probabilities:"
)


for label, probability in zip(
        model.classes_,
        new_probabilities[0]
):

    print(
        label,
        ":",
        round(probability, 4)
    )

print()


# ==================================================
# 17. MODEL CONFIDENCE
# ==================================================

confidence = np.max(
    new_probabilities[0]
)


print(
    "Highest model probability:",
    round(confidence, 4)
)

print()


# ==================================================
# 18. TRAINING PERFORMANCE
# ==================================================

print("TEST 16: Training Performance")
print()


train_predictions = model.predict(
    X_train
)


train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


print(
    "Training accuracy:",
    train_accuracy
)

print()


# ==================================================
# 19. GENERALIZATION CHECK
# ==================================================

print("TEST 17: Generalization")
print()


print(
    "Training accuracy:",
    train_accuracy
)

print(
    "Testing accuracy:",
    accuracy
)


accuracy_gap = (
        train_accuracy
        -
        accuracy
)


print(
    "Accuracy gap:",
    accuracy_gap
)

print()


if accuracy_gap > 0.20:

    print(
        "Possible overfitting signal."
    )

else:

    print(
        "No large accuracy gap detected."
    )

print()


# ==================================================
# 20. SAVE RESULTS
# ==================================================

print("TEST 18: Save Predictions")
print()


results.to_csv(
    "classification_predictions.csv",
    index=False
)


probability_df.to_csv(
    "classification_probabilities.csv",
    index=False
)


print(
    "Saved classification_predictions.csv"
)

print(
    "Saved classification_probabilities.csv"
)

print()


# ==================================================
# 21. ML CLASSIFICATION PIPELINE
# ==================================================

print("CLASSIFICATION PIPELINE")
print()

print("Machine data")
print("      ↓")
print("Feature selection")
print("      ↓")
print("Train / test split")
print("      ↓")
print("Logistic Regression")
print("      ↓")
print("Training")
print("      ↓")
print("Class prediction")
print("      ↓")
print("Probability")
print("      ↓")
print("Evaluation")

print()


# ==================================================
# 22. REGRESSION VS CLASSIFICATION
# ==================================================

print("REGRESSION VS CLASSIFICATION")
print()

print(
    "Regression predicts continuous values."
)

print(
    "Example: risk score = 63.4"
)

print()

print(
    "Classification predicts categories."
)

print(
    "Example: risk level = HIGH"
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 23 COMPLETE ===")
