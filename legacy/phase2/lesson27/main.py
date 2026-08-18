# Silverwing ML
# Phase 2 - Lesson 27
# Cross-Validation and Hyperparameter Tuning


import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 27")
print("Cross-Validation and Hyperparameter Tuning")
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
# 2. DISPLAY DATASET INFORMATION
# ==================================================

print("TEST 1: Dataset")
print()

print(
    "Number of observations:",
    len(df)
)

print(
    "Number of features:",
    4
)

print()

print(df)

print()


# ==================================================
# 3. FEATURES AND TARGET
# ==================================================

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]

X = df[feature_columns]

y = df["risk_level"]


print("TEST 2: Features and Target")
print()

print("Features:")
print(X)

print()

print("Target:")
print(y)

print()


# ==================================================
# 4. CLASS DISTRIBUTION
# ==================================================

print("TEST 3: Class Distribution")
print()

class_counts = y.value_counts()

print(class_counts)

print()


# ==================================================
# 5. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("TEST 4: Train/Test Split")
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
# 6. CROSS-VALIDATION PIPELINE
# ==================================================

print("TEST 5: Cross-Validation")
print()


logistic_pipeline = Pipeline([
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


# Four folds are used because this small teaching
# dataset has only four examples in its smallest class.

cv_folds = 4


cv_scores = cross_val_score(
    logistic_pipeline,
    X_train,
    y_train,
    cv=cv_folds,
    scoring="accuracy"
)


print(
    "Number of CV folds:",
    cv_folds
)

print()

print(
    "Cross-validation scores:"
)

print(cv_scores)

print()

print(
    "Mean CV accuracy:",
    cv_scores.mean()
)

print(
    "CV standard deviation:",
    cv_scores.std()
)

print()


# ==================================================
# 7. CROSS-VALIDATION INTERPRETATION
# ==================================================

print("TEST 6: Cross-Validation Interpretation")
print()

print(
    "Cross-validation evaluates a model across "
    "multiple training and validation splits."
)

print()

print(
    "The mean score summarizes performance "
    "across those validation splits."
)

print()

print(
    "The standard deviation shows how much "
    "the validation scores vary between folds."
)

print()


# ==================================================
# 8. CREATE RANDOM FOREST
# ==================================================

print("TEST 7: Hyperparameter Search Setup")
print()


forest = RandomForestClassifier(
    random_state=42
)


parameter_grid = {
    "n_estimators": [
        25,
        50,
        100
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        None
    ],

    "min_samples_split": [
        2,
        4
    ]
}


print(
    "Hyperparameters to test:"
)

for parameter, values in parameter_grid.items():

    print(
        parameter,
        ":",
        values
    )

print()


# ==================================================
# 9. CALCULATE NUMBER OF CONFIGURATIONS
# ==================================================

number_of_configurations = 1

for values in parameter_grid.values():

    number_of_configurations *= len(values)


print(
    "Model configurations:",
    number_of_configurations
)

print(
    "Validation folds per configuration:",
    cv_folds
)

print(
    "Total model fits:",
    number_of_configurations * cv_folds
)

print()


# ==================================================
# 10. GRID SEARCH
# ==================================================

print("TEST 8: Grid Search")
print()

grid_search = GridSearchCV(
    estimator=forest,
    param_grid=parameter_grid,
    cv=cv_folds,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True
)


print(
    "Starting grid search..."
)

print()


grid_search.fit(
    X_train,
    y_train
)


print(
    "Grid search complete."
)

print()


# ==================================================
# 11. BEST PARAMETERS
# ==================================================

print("TEST 9: Best Parameters")
print()

best_parameters = (
    grid_search.best_params_
)


print(
    "Best parameters:"
)

print(
    best_parameters
)

print()


# ==================================================
# 12. BEST CROSS-VALIDATION SCORE
# ==================================================

best_cv_score = (
    grid_search.best_score_
)


print(
    "Best cross-validation accuracy:"
)

print(
    best_cv_score
)

print()


# ==================================================
# 13. BEST MODEL
# ==================================================

print("TEST 10: Best Model")
print()

best_model = (
    grid_search.best_estimator_
)


print(
    best_model
)

print()


# ==================================================
# 14. GRID SEARCH RESULTS
# ==================================================

print("TEST 11: Top Model Configurations")
print()


cv_results = pd.DataFrame(
    grid_search.cv_results_
)


top_results = cv_results[
    [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "mean_test_score",
        "std_test_score",
        "mean_train_score"
    ]
].sort_values(
    by="mean_test_score",
    ascending=False
)


print(
    top_results.head(10)
)

print()


# ==================================================
# 15. TEST SET PREDICTIONS
# ==================================================

print("TEST 12: Test Predictions")
print()


test_predictions = (
    best_model.predict(X_test)
)


prediction_results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": test_predictions
})


print(
    prediction_results
)

print()


# ==================================================
# 16. TEST ACCURACY
# ==================================================

print("TEST 13: Test Accuracy")
print()


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print(
    "Accuracy:",
    test_accuracy
)

print(
    "Percentage:",
    test_accuracy * 100,
    "%"
)

print()


# ==================================================
# 17. TEST PRECISION
# ==================================================

test_precision = precision_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


print("TEST 14: Test Precision")
print()

print(
    "Weighted precision:",
    test_precision
)

print()


# ==================================================
# 18. TEST RECALL
# ==================================================

test_recall = recall_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


print("TEST 15: Test Recall")
print()

print(
    "Weighted recall:",
    test_recall
)

print()


# ==================================================
# 19. TEST F1
# ==================================================

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)


print("TEST 16: Test F1")
print()

print(
    "Weighted F1:",
    test_f1
)

print()


# ==================================================
# 20. CLASSIFICATION REPORT
# ==================================================

print("TEST 17: Classification Report")
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
# 21. TRAINING PERFORMANCE
# ==================================================

print("TEST 18: Training Performance")
print()


train_predictions = (
    best_model.predict(X_train)
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
    test_accuracy
)

print()


# ==================================================
# 22. GENERALIZATION CHECK
# ==================================================

print("TEST 19: Generalization Check")
print()


accuracy_gap = (
        train_accuracy
        -
        test_accuracy
)


print(
    "Training accuracy:",
    train_accuracy
)

print(
    "Testing accuracy:",
    test_accuracy
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
        "No large training/testing accuracy gap "
        "detected by this simple check."
    )

print()


# ==================================================
# 23. NEW MACHINE
# ==================================================

print("TEST 20: New Machine Prediction")
print()


new_machine = pd.DataFrame({
    "temperature": [97],
    "pressure": [130],
    "rpm": [2600],
    "operating_hours": [3500]
})


new_prediction = (
    best_model.predict(
        new_machine
    )
)


print(
    "New machine:"
)

print(
    new_machine
)

print()

print(
    "Predicted risk level:",
    new_prediction[0]
)

print()


# ==================================================
# 24. NEW MACHINE PROBABILITIES
# ==================================================

print("TEST 21: Class Probabilities")
print()


new_probability = (
    best_model.predict_proba(
        new_machine
    )
)


print(
    "Model classes:"
)

print(
    best_model.classes_
)

print()


for label, probability in zip(
        best_model.classes_,
        new_probability[0]
):

    print(
        label,
        ":",
        round(
            probability,
            4
        )
    )

print()


# ==================================================
# 25. MODEL CONFIDENCE
# ==================================================

confidence = np.max(
    new_probability[0]
)


print(
    "Highest probability:",
    round(
        confidence,
        4
    )
)

print()


# ==================================================
# 26. FEATURE IMPORTANCE
# ==================================================

print("TEST 22: Feature Importance")
print()


importance = pd.DataFrame({
    "feature": feature_columns,
    "importance":
        best_model.feature_importances_
})


importance = importance.sort_values(
    by="importance",
    ascending=False
)


print(
    importance
)

print()


# ==================================================
# 27. SAVE HYPERPARAMETER RESULTS
# ==================================================

print("TEST 23: Save Search Results")
print()


cv_results.to_csv(
    "hyperparameter_search_results.csv",
    index=False
)


print(
    "Saved:"
)

print(
    "hyperparameter_search_results.csv"
)

print()


# ==================================================
# 28. SAVE FEATURE IMPORTANCE
# ==================================================

print("TEST 24: Save Feature Importance")
print()


importance.to_csv(
    "tuned_model_feature_importance.csv",
    index=False
)


print(
    "Saved:"
)

print(
    "tuned_model_feature_importance.csv"
)

print()


# ==================================================
# 29. SAVE PREDICTIONS
# ==================================================

print("TEST 25: Save Predictions")
print()


prediction_results.to_csv(
    "tuned_model_predictions.csv",
    index=False
)


print(
    "Saved:"
)

print(
    "tuned_model_predictions.csv"
)

print()


# ==================================================
# 30. SAVE MODEL SUMMARY
# ==================================================

print("TEST 26: Model Summary")
print()


model_summary = {
    "model": "RandomForestClassifier",
    "cross_validation_folds": cv_folds,
    "best_cv_accuracy": best_cv_score,
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1": test_f1,
    "training_accuracy": train_accuracy,
    "accuracy_gap": accuracy_gap
}


summary_df = pd.DataFrame([
    model_summary
])


summary_df.to_csv(
    "tuned_model_summary.csv",
    index=False
)


print(
    summary_df.to_string(
        index=False
    )
)

print()


# ==================================================
# 31. EXPLAIN CROSS-VALIDATION
# ==================================================

print("CROSS-VALIDATION")
print()

print(
    "Training data is divided into multiple folds."
)

print()

print(
    "Each fold takes a turn being the validation data."
)

print()

print(
    "The scores are then averaged."
)

print()


# ==================================================
# 32. EXPLAIN HYPERPARAMETERS
# ==================================================

print("HYPERPARAMETERS")
print()

print(
    "Hyperparameters are settings chosen "
    "before or during model training."
)

print()

print(
    "Examples used in this lesson:"
)

print(
    "- Number of trees"
)

print(
    "- Maximum tree depth"
)

print(
    "- Minimum samples required to split"
)

print()


# ==================================================
# 33. WHY THE TEST SET IS KEPT SEPARATE
# ==================================================

print("FINAL VALIDATION PRINCIPLE")
print()

print(
    "Cross-validation and hyperparameter search "
    "use the training portion."
)

print()

print(
    "The held-out test set is kept separate "
    "for final evaluation."
)

print()

print(
    "This helps provide a less biased estimate "
    "of performance on unseen data."
)

print()


# ==================================================
# 34. CURRENT SILVERWING ML PIPELINE
# ==================================================

print("CURRENT SILVERWING ML PIPELINE")
print()

print("Raw data")
print("   ↓")
print("Cleaning")
print("   ↓")
print("Features and target")
print("   ↓")
print("Train/Test split")
print("   ↓")
print("Cross-validation")
print("   ↓")
print("Hyperparameter search")
print("   ↓")
print("Best model")
print("   ↓")
print("Final test evaluation")
print("   ↓")
print("New machine prediction")

print()


# ==================================================
# 35. IMPORTANT DATASET NOTE
# ==================================================

print("DATASET NOTE")
print()

print(
    "This is a small educational dataset."
)

print()

print(
    "Its scores should not be interpreted as "
    "evidence of real-world predictive performance."
)

print()

print(
    "A production ML system would require "
    "larger and representative historical data, "
    "careful validation, and domain-specific testing."
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 27 COMPLETE ===")
