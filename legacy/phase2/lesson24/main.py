# Silverwing ML
# Phase 2 - Lesson 24
# Decision Trees
# Machine Risk Classification

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    export_text
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 24")
print("Decision Trees")
print()


# ==================================================
# 1. CREATE DATASET
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
# 2. DISPLAY DATA
# ==================================================

print("TEST 1: Dataset")
print()

print(df)

print()


# ==================================================
# 3. DEFINE FEATURES AND TARGET
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]

X = df[feature_columns]
y = df["risk_level"]


print("TEST 2: Features")
print()

print(X)

print()

print("Target:")
print(y)

print()


# ==================================================
# 4. CHECK CLASS DISTRIBUTION
# ==================================================

print("TEST 3: Class Distribution")
print()

print(y.value_counts())

print()


# ==================================================
# 5. SPLIT DATA
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


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print()


# ==================================================
# 6. CREATE DECISION TREE
# ==================================================

print("TEST 5: Create Decision Tree")
print()


model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


print("Decision tree created.")
print()


# ==================================================
# 7. TRAIN MODEL
# ==================================================

print("TEST 6: Train Decision Tree")
print()


model.fit(
    X_train,
    y_train
)


print("Decision tree training complete.")
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


print("Accuracy:", accuracy)
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


print("Labels:")
print(labels)

print()

print(matrix)

print()


# ==================================================
# 15. FEATURE IMPORTANCE
# ==================================================

print("TEST 14: Feature Importance")
print()


importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})


importance = importance.sort_values(
    by="importance",
    ascending=False
)


print(importance)

print()


# ==================================================
# 16. DISPLAY LEARNED TREE RULES
# ==================================================

print("TEST 15: Learned Tree Rules")
print()


tree_rules = export_text(
    model,
    feature_names=feature_columns
)


print(tree_rules)

print()


# ==================================================
# 17. PREDICT A NEW MACHINE
# ==================================================

print("TEST 16: New Machine Prediction")
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


print("New machine:")
print(new_machine)

print()

print(
    "Predicted risk level:",
    new_prediction[0]
)

print()


# ==================================================
# 18. PREDICT PROBABILITIES
# ==================================================

print("TEST 17: Class Probabilities")
print()


probabilities = model.predict_proba(
    new_machine
)


print("Model classes:")
print(model.classes_)

print()

for label, probability in zip(
        model.classes_,
        probabilities[0]
):

    print(
        label,
        ":",
        round(probability, 4)
    )

print()


# ==================================================
# 19. TRAINING PERFORMANCE
# ==================================================

print("TEST 18: Training Performance")
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

print(
    "Testing accuracy:",
    accuracy
)

print()


# ==================================================
# 20. SIMPLE OVERFITTING CHECK
# ==================================================

accuracy_gap = (
        train_accuracy
        -
        accuracy
)


print("TEST 19: Generalization Check")
print()

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
        "No large training/testing accuracy gap."
    )

print()


# ==================================================
# 21. SAVE PREDICTIONS
# ==================================================

print("TEST 20: Save Results")
print()


results.to_csv(
    "decision_tree_predictions.csv",
    index=False
)


importance.to_csv(
    "decision_tree_feature_importance.csv",
    index=False
)


print(
    "Saved decision_tree_predictions.csv"
)

print(
    "Saved decision_tree_feature_importance.csv"
)

print()


# ==================================================
# 22. MACHINE LEARNING CONCEPT
# ==================================================

print("DECISION TREE CONCEPT")
print()

print(
    "A decision tree learns a sequence "
    "of feature-based decisions from data."
)

print()

print(
    "Instead of manually writing:"
)

print(
    "temperature > threshold -> WARNING"
)

print()

print(
    "the algorithm learns useful thresholds "
    "from the training examples."
)

print()


# ==================================================
# 23. CURRENT ML PIPELINE
# ==================================================

print("CURRENT SILVERWING ML PIPELINE")
print()

print("Data")
print(" ↓")
print("Cleaning")
print(" ↓")
print("Features / Target")
print(" ↓")
print("Train / Test Split")
print(" ↓")
print("Decision Tree")
print(" ↓")
print("Training")
print(" ↓")
print("Prediction")
print(" ↓")
print("Evaluation")
print(" ↓")
print("Feature Importance")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 24 COMPLETE ===")
