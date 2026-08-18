# Silverwing ML
# Phase 2 - Lesson 28
# Saving and Loading Machine Learning Models


import os

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 28")
print("Saving and Loading Machine Learning Models")
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
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",

        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",

        "CRITICAL",
        "CRITICAL",
        "CRITICAL",
        "CRITICAL",
        "CRITICAL"
    ]
}


df = pd.DataFrame(data)


# ==================================================
# 2. DEFINE FEATURES AND TARGET
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]

X = df[feature_columns]
y = df["risk_level"]


print("TEST 1: Dataset")
print()

print(
    "Observations:",
    len(df)
)

print(
    "Features:",
    feature_columns
)

print()


# ==================================================
# 3. SPLIT DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("TEST 2: Train/Test Split")
print()

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
# 4. CREATE MODEL
# ==================================================

print("TEST 3: Create Model")
print()


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


print(
    "Model:",
    type(model).__name__
)

print()


# ==================================================
# 5. TRAIN MODEL
# ==================================================

print("TEST 4: Train Model")
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
# 6. EVALUATE MODEL
# ==================================================

print("TEST 5: Evaluate Model")
print()


test_predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    test_predictions
)


print(
    "Test accuracy:",
    accuracy
)

print()

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print()


# ==================================================
# 7. PREDICT BEFORE SAVING
# ==================================================

print("TEST 6: Prediction Before Saving")
print()


new_machine = pd.DataFrame({
    "temperature": [97],
    "pressure": [130],
    "rpm": [2600],
    "operating_hours": [3500]
})


prediction_before_save = model.predict(
    new_machine
)


probability_before_save = model.predict_proba(
    new_machine
)


print("New machine:")
print(new_machine)

print()

print(
    "Prediction:",
    prediction_before_save[0]
)

print()

print("Probabilities:")

for label, probability in zip(
        model.classes_,
        probability_before_save[0]
):

    print(
        label,
        ":",
        round(probability, 4)
    )

print()


# ==================================================
# 8. SAVE MODEL
# ==================================================

print("TEST 7: Save Model")
print()


model_file = "silverwing_risk_model.joblib"


joblib.dump(
    model,
    model_file
)


print(
    "Model saved as:",
    model_file
)

print()


# ==================================================
# 9. VERIFY MODEL FILE
# ==================================================

print("TEST 8: Verify Saved Model")
print()


if os.path.exists(model_file):

    file_size = os.path.getsize(
        model_file
    )

    print(
        "Model file exists."
    )

    print(
        "File size:",
        file_size,
        "bytes"
    )

else:

    print(
        "ERROR: Model file was not created."
    )

print()


# ==================================================
# 10. DELETE MODEL FROM MEMORY
# ==================================================

print("TEST 9: Simulate Application Restart")
print()


del model


print(
    "Original model removed from memory."
)

print()


# ==================================================
# 11. LOAD MODEL
# ==================================================

print("TEST 10: Load Saved Model")
print()


loaded_model = joblib.load(
    model_file
)


print(
    "Saved model successfully loaded."
)

print(
    "Loaded model type:",
    type(loaded_model).__name__
)

print()


# ==================================================
# 12. PREDICT WITH LOADED MODEL
# ==================================================

print("TEST 11: Prediction After Loading")
print()


prediction_after_load = (
    loaded_model.predict(
        new_machine
    )
)


probability_after_load = (
    loaded_model.predict_proba(
        new_machine
    )
)


print(
    "Prediction from loaded model:",
    prediction_after_load[0]
)

print()

print(
    "Probabilities from loaded model:"
)

for label, probability in zip(
        loaded_model.classes_,
        probability_after_load[0]
):

    print(
        label,
        ":",
        round(probability, 4)
    )

print()


# ==================================================
# 13. VERIFY PREDICTION CONSISTENCY
# ==================================================

print("TEST 12: Verify Model Consistency")
print()


prediction_matches = (
        prediction_before_save[0]
        ==
        prediction_after_load[0]
)


print(
    "Prediction before saving:",
    prediction_before_save[0]
)

print(
    "Prediction after loading:",
    prediction_after_load[0]
)

print(
    "Predictions match:",
    prediction_matches
)

print()


if prediction_matches:

    print(
        "Model persistence test PASSED."
    )

else:

    print(
        "Model persistence test FAILED."
    )

print()


# ==================================================
# 14. MODEL METADATA
# ==================================================

print("TEST 13: Model Metadata")
print()


metadata = {
    "model_name": "Silverwing Risk Classifier",
    "algorithm": "RandomForestClassifier",
    "n_estimators": 100,
    "max_depth": 5,
    "features": feature_columns,
    "target": "risk_level",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "test_accuracy": float(accuracy)
}


metadata_file = "silverwing_model_metadata.json"


import json

with open(
        metadata_file,
        "w",
        encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print(
    "Metadata saved as:",
    metadata_file
)

print()


# ==================================================
# 15. LOAD METADATA
# ==================================================

print("TEST 14: Load Metadata")
print()


with open(
        metadata_file,
        "r",
        encoding="utf-8"
) as file:

    loaded_metadata = json.load(
        file
    )


for key, value in loaded_metadata.items():

    print(
        key,
        ":",
        value
    )

print()


# ==================================================
# 16. MODEL FILES
# ==================================================

print("TEST 15: Model Artifacts")
print()


print(
    "Model artifact:",
    model_file
)

print(
    "Metadata artifact:",
    metadata_file
)

print()


# ==================================================
# 17. MODEL SERVICE CONCEPT
# ==================================================

print("MODEL SERVICE CONCEPT")
print()

print(
    "A running AI service can load the "
    "trained model once."
)

print()

print(
    "It can then receive new machine data "
    "and make predictions without retraining."
)

print()


# ==================================================
# 18. FUTURE SILVERWING ARCHITECTURE
# ==================================================

print("FUTURE SILVERWING ARCHITECTURE")
print()

print("User / Sensor")
print("     ↓")
print("Silverwing API")
print("     ↓")
print("ML Model Service")
print("     ↓")
print("Loaded Model")
print("     ↓")
print("Prediction")
print("     ↓")
print("AI Reasoning")
print("     ↓")
print("Communicative Response")

print()


# ==================================================
# 19. WHY MODEL PERSISTENCE MATTERS
# ==================================================

print("WHY MODEL PERSISTENCE MATTERS")
print()

print(
    "Training can be expensive."
)

print()

print(
    "Inference is usually performed using "
    "an already-trained model."
)

print()

print(
    "Saving the model allows applications "
    "to reuse the learned parameters."
)

print()


# ==================================================
# 20. CURRENT ML PIPELINE
# ==================================================

print("CURRENT SILVERWING ML PIPELINE")
print()

print("Historical Data")
print("      ↓")
print("Cleaning")
print("      ↓")
print("Feature Engineering")
print("      ↓")
print("Model Training")
print("      ↓")
print("Model Evaluation")
print("      ↓")
print("Save Model")
print("      ↓")
print("Load Model")
print("      ↓")
print("Prediction")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 28 COMPLETE ===")
