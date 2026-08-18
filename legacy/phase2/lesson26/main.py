# Silverwing ML
# Phase 2 - Lesson 26
# Feature Scaling and ML Pipelines

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 26")
print("Feature Scaling and ML Pipelines")
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
# 2. FEATURES AND TARGET
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]

X = df[feature_columns]
y = df["risk_level"]


print("TEST 1: Original Feature Ranges")
print()

for feature in feature_columns:

    print(
        feature,
        "->",
        "min:",
        X[feature].min(),
        "| max:",
        X[feature].max()
    )

print()


# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================

print("TEST 2: Train/Test Split")
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
# 4. CREATE STANDARD SCALER
# ==================================================

print("TEST 3: StandardScaler")
print()

scaler = StandardScaler()

print(
    "Scaler created:",
    type(scaler).__name__
)

print()


# ==================================================
# 5. FIT SCALER ONLY ON TRAINING DATA
# ==================================================

print("TEST 4: Fit Scaler on Training Data")
print()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print(
    "Training data scaled."
)

print(
    "Testing data scaled using training parameters."
)

print()


# ==================================================
# 6. DISPLAY SCALED DATA
# ==================================================

print("TEST 5: Scaled Training Data")
print()

scaled_training_df = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

print(
    scaled_training_df.round(3)
)

print()


# ==================================================
# 7. INSPECT SCALE STATISTICS
# ==================================================

print("TEST 6: Scaled Feature Statistics")
print()

print(
    scaled_training_df.describe().round(3)
)

print()


# ==================================================
# 8. TRAIN MODEL WITHOUT PIPELINE
# ==================================================

print("TEST 7: Train Logistic Regression")
print()

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "Model trained on scaled data."
)

print()


# ==================================================
# 9. MAKE PREDICTIONS
# ==================================================

print("TEST 8: Predictions")
print()

predictions = model.predict(
    X_test_scaled
)

print(
    predictions
)

print()


# ==================================================
# 10. EVALUATE
# ==================================================

print("TEST 9: Model Evaluation")
print()

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print()


# ==================================================
# 11. CLASSIFICATION REPORT
# ==================================================

print("TEST 10: Classification Report")
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
# 12. BUILD A PIPELINE
# ==================================================

print("TEST 11: Build ML Pipeline")
print()

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000
        )
    )
])


print(
    "Pipeline created."
)

print()


# ==================================================
# 13. TRAIN PIPELINE
# ==================================================

print("TEST 12: Train Pipeline")
print()

pipeline.fit(
    X_train,
    y_train
)

print(
    "Pipeline training complete."
)

print()


# ==================================================
# 14. PIPELINE PREDICTIONS
# ==================================================

print("TEST 13: Pipeline Predictions")
print()

pipeline_predictions = pipeline.predict(
    X_test
)

print(
    pipeline_predictions
)

print()


# ==================================================
# 15. PIPELINE EVALUATION
# ==================================================

print("TEST 14: Pipeline Evaluation")
print()

pipeline_accuracy = accuracy_score(
    y_test,
    pipeline_predictions
)

pipeline_f1 = f1_score(
    y_test,
    pipeline_predictions,
    average="weighted",
    zero_division=0
)

print(
    "Pipeline accuracy:",
    pipeline_accuracy
)

print(
    "Pipeline F1:",
    pipeline_f1
)

print()


# ==================================================
# 16. COMPARE MANUAL VS PIPELINE
# ==================================================

print("TEST 15: Compare Approaches")
print()

print(
    "Manual scaling accuracy:",
    accuracy
)

print(
    "Pipeline accuracy:",
    pipeline_accuracy
)

print()

print(
    "Manual scaling F1:",
    f1
)

print(
    "Pipeline F1:",
    pipeline_f1
)

print()


# ==================================================
# 17. NEW MACHINE
# ==================================================

print("TEST 16: New Machine Prediction")
print()

new_machine = pd.DataFrame({
    "temperature": [97],
    "pressure": [130],
    "rpm": [2600],
    "operating_hours": [3500]
})


new_prediction = pipeline.predict(
    new_machine
)

new_probability = pipeline.predict_proba(
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

print("Class probabilities:")

for label, probability in zip(
        pipeline.classes_,
        new_probability[0]
):

    print(
        label,
        ":",
        round(probability, 4)
    )

print()


# ==================================================
# 18. PIPELINE STRUCTURE
# ==================================================

print("TEST 17: Pipeline Structure")
print()

print(
    pipeline
)

print()


# ==================================================
# 19. WHY PIPELINES MATTER
# ==================================================

print("WHY PIPELINES MATTER")
print()

print(
    "A pipeline keeps preprocessing and "
    "model training together."
)

print()

print(
    "It reduces the chance of applying "
    "different preprocessing during training "
    "and prediction."
)

print()

print(
    "It also makes the ML workflow easier "
    "to reproduce and deploy."
)

print()


# ==================================================
# 20. IMPORTANT DATA-LEAKAGE RULE
# ==================================================

print("DATA LEAKAGE RULE")
print()

print(
    "The scaler must learn its parameters "
    "from training data only."
)

print()

print(
    "Test data must remain unseen during "
    "the fitting process."
)

print()


# ==================================================
# 21. CURRENT ML PIPELINE
# ==================================================

print("CURRENT SILVERWING ML PIPELINE")
print()

print("Raw data")
print(" ↓")
print("Cleaning")
print(" ↓")
print("Train / Test split")
print(" ↓")
print("Feature scaling")
print(" ↓")
print("ML model")
print(" ↓")
print("Prediction")
print(" ↓")
print("Evaluation")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 26 COMPLETE ===")
