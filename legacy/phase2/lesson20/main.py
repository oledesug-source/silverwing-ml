# Silverwing ML
# Phase 2 - Lesson 20
# Training and Testing Data

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 20")
print("Training and Testing Data")
print()


# ==================================================
# 1. CREATE MACHINE DATA
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
# 3. SELECT FEATURES
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]


X = df[feature_columns]

y = df["risk_score"]


print("TEST 2: Features and Target")
print()

print("Features:")
print(X)

print()

print("Target:")
print(y)

print()


# ==================================================
# 4. CHECK DATASET SIZE
# ==================================================

print("TEST 3: Dataset Size")
print()

print("Total observations:", len(df))
print("Total features:", X.shape[1])

print()


# ==================================================
# 5. SPLIT THE DATA
# ==================================================

print("TEST 4: Train/Test Split")
print()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print()


# ==================================================
# 6. DISPLAY TRAINING DATA
# ==================================================

print("TEST 5: Training Features")
print()

print(X_train)

print()

print("Training targets:")
print(y_train)

print()


# ==================================================
# 7. DISPLAY TESTING DATA
# ==================================================

print("TEST 6: Testing Features")
print()

print(X_test)

print()

print("Testing targets:")
print(y_test)

print()


# ==================================================
# 8. VERIFY SPLIT
# ==================================================

print("TEST 7: Verify Dataset Split")
print()

total_samples = (
        len(X_train)
        +
        len(X_test)
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Combined samples:", total_samples)
print("Original samples:", len(df))

print()


if total_samples == len(df):

    print("Dataset split verified.")

else:

    print("Dataset split verification failed.")

print()


# ==================================================
# 9. CHECK FOR OVERLAPPING INDEXES
# ==================================================

print("TEST 8: Check for Data Leakage")
print()

training_indexes = set(
    X_train.index
)

testing_indexes = set(
    X_test.index
)


overlap = (
        training_indexes
        &
        testing_indexes
)


print(
    "Overlapping observations:",
    overlap
)


if len(overlap) == 0:

    print(
        "No overlapping observations detected."
    )

else:

    print(
        "Potential data leakage detected."
    )

print()


# ==================================================
# 10. RESET INDEXES
# ==================================================

print("TEST 9: Reset Indexes")
print()

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


print("Training data:")
print(X_train)

print()

print("Testing data:")
print(X_test)

print()


# ==================================================
# 11. SAVE TRAINING DATA
# ==================================================

print("TEST 10: Save Training Data")
print()

training_data = X_train.copy()

training_data["risk_score"] = y_train


training_data.to_csv(
    "training_data.csv",
    index=False
)


print(
    "Training data saved as training_data.csv"
)

print()


# ==================================================
# 12. SAVE TESTING DATA
# ==================================================

print("TEST 11: Save Testing Data")
print()

testing_data = X_test.copy()

testing_data["risk_score"] = y_test


testing_data.to_csv(
    "testing_data.csv",
    index=False
)


print(
    "Testing data saved as testing_data.csv"
)

print()


# ==================================================
# 13. EXPLAIN THE PIPELINE
# ==================================================

print("MACHINE LEARNING WORKFLOW")
print()

print("1. Collect data")
print("2. Clean data")
print("3. Separate features and target")
print("4. Split training and testing data")
print("5. Train model using training data")
print("6. Test model using unseen testing data")
print("7. Measure performance")
print("8. Improve the model")

print()


# ==================================================
# 14. WHY TEST DATA IS IMPORTANT
# ==================================================

print("WHY TEST DATA MATTERS")
print()

print(
    "The model should learn from the training data."
)

print()

print(
    "The testing data represents observations "
    "the model did not use during training."
)

print()

print(
    "This helps us estimate how well the model "
    "may perform on unseen data."
)

print()


# ==================================================
# 15. CURRENT PROJECT STATE
# ==================================================

print("CURRENT SILVERWING ML PIPELINE")
print()

print("Raw machine data")
print("        ↓")
print("Data cleaning")
print("        ↓")
print("Feature preparation")
print("        ↓")
print("Train / test split")
print("        ↓")
print("READY FOR MODEL TRAINING")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 20 COMPLETE ===")
