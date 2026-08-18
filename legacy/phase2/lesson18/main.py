# Silverwing ML
# Phase 2 - Lesson 18
# Pandas and DataFrames

import pandas as pd
import numpy as np


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 18")
print("Pandas and DataFrames")
print()


# ==================================================
# 1. CREATE MACHINE DATA
# ==================================================

data = {
    "machine": [
        "Pump",
        "Compressor",
        "Generator",
        "Turbine",
        "Boiler"
    ],
    "temperature": [
        85,
        72,
        105,
        91,
        98
    ],
    "pressure": [
        120,
        150,
        110,
        135,
        145
    ],
    "rpm": [
        1500,
        2800,
        3200,
        2900,
        2600
    ],
    "operating_hours": [
        2500,
        3200,
        4500,
        3800,
        4200
    ]
}


# ==================================================
# 2. CREATE DATAFRAME
# ==================================================

print("TEST 1: Create DataFrame")
print()

df = pd.DataFrame(data)

print(df)
print()


# ==================================================
# 3. INSPECT DATAFRAME
# ==================================================

print("TEST 2: DataFrame Shape")
print()

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print()


print("TEST 3: Column Names")
print()

print(df.columns.tolist())
print()


# ==================================================
# 4. SELECT ONE COLUMN
# ==================================================

print("TEST 4: Select Temperature")
print()

print(df["temperature"])
print()


# ==================================================
# 5. SELECT MULTIPLE COLUMNS
# ==================================================

print("TEST 5: Select Multiple Columns")
print()

print(
    df[
        [
            "machine",
            "temperature",
            "rpm"
        ]
    ]
)

print()


# ==================================================
# 6. SELECT A SINGLE ROW
# ==================================================

print("TEST 6: Select First Row")
print()

print(df.iloc[0])
print()


# ==================================================
# 7. SELECT MULTIPLE ROWS
# ==================================================

print("TEST 7: Select First Three Rows")
print()

print(df.iloc[0:3])
print()


# ==================================================
# 8. FILTER HIGH TEMPERATURE MACHINES
# ==================================================

print("TEST 8: High Temperature Machines")
print()

high_temperature = df[
    df["temperature"] >= 90
    ]

print(high_temperature)
print()


# ==================================================
# 9. FILTER HIGH RPM MACHINES
# ==================================================

print("TEST 9: High RPM Machines")
print()

high_rpm = df[
    df["rpm"] > 3000
    ]

print(high_rpm)
print()


# ==================================================
# 10. TEMPERATURE STATUS
# ==================================================

def temperature_status(temperature):

    if temperature >= 100:
        return "CRITICAL"

    elif temperature >= 80:
        return "HIGH"

    else:
        return "NORMAL"


df["temperature_status"] = df[
    "temperature"
].apply(
    temperature_status
)


print("TEST 10: Temperature Status")
print()

print(
    df[
        [
            "machine",
            "temperature",
            "temperature_status"
        ]
    ]
)

print()


# ==================================================
# 11. CALCULATE RISK SCORE
# ==================================================

def calculate_risk(row):

    score = 0

    temperature = row["temperature"]
    pressure = row["pressure"]
    rpm = row["rpm"]

    if temperature >= 100:
        score += 40

    elif temperature >= 80:
        score += 20

    if rpm > 3000:
        score += 40

    elif rpm > 2500:
        score += 15

    if pressure >= 160:
        score += 20

    elif pressure >= 130:
        score += 10

    return score


df["risk_score"] = df.apply(
    calculate_risk,
    axis=1
)


print("TEST 11: Risk Scores")
print()

print(
    df[
        [
            "machine",
            "risk_score"
        ]
    ]
)

print()


# ==================================================
# 12. CLASSIFY RISK
# ==================================================

def risk_level(score):

    if score >= 70:
        return "CRITICAL"

    elif score >= 40:
        return "HIGH"

    elif score >= 20:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_level"] = df[
    "risk_score"
].apply(
    risk_level
)


print("TEST 12: Risk Classification")
print()

print(
    df[
        [
            "machine",
            "risk_score",
            "risk_level"
        ]
    ]
)

print()


# ==================================================
# 13. DESCRIPTIVE STATISTICS
# ==================================================

print("TEST 13: Descriptive Statistics")
print()

print(df.describe())

print()


# ==================================================
# 14. SENSOR AVERAGES
# ==================================================

print("TEST 14: Sensor Averages")
print()

print(
    "Average temperature:",
    df["temperature"].mean()
)

print(
    "Average pressure:",
    df["pressure"].mean()
)

print(
    "Average RPM:",
    df["rpm"].mean()
)

print(
    "Average operating hours:",
    df["operating_hours"].mean()
)

print()


# ==================================================
# 15. SORT BY RISK
# ==================================================

print("TEST 15: Sort Machines by Risk")
print()

sorted_df = df.sort_values(
    by="risk_score",
    ascending=False
)

print(
    sorted_df[
        [
            "machine",
            "risk_score",
            "risk_level"
        ]
    ]
)

print()


# ==================================================
# 16. FIND HIGHEST-RISK MACHINE
# ==================================================

print("TEST 16: Highest-Risk Machine")
print()

highest_risk_index = df[
    "risk_score"
].idxmax()

highest_risk_machine = df.loc[
    highest_risk_index
]

print(
    "Machine:",
    highest_risk_machine["machine"]
)

print(
    "Risk score:",
    highest_risk_machine["risk_score"]
)

print(
    "Risk level:",
    highest_risk_machine["risk_level"]
)

print()


# ==================================================
# 17. CREATE MISSING DATA EXAMPLE
# ==================================================

print("TEST 17: Missing Data")
print()

df_with_missing = df.copy()

df_with_missing.loc[
    2,
    "pressure"
] = np.nan

print(df_with_missing)

print()

print("Missing values by column:")

print(
    df_with_missing.isnull().sum()
)

print()


# ==================================================
# 18. FILL MISSING DATA
# ==================================================

print("TEST 18: Fill Missing Data")
print()

pressure_mean = df_with_missing[
    "pressure"
].mean()

df_with_missing[
    "pressure"
] = df_with_missing[
    "pressure"
].fillna(
    pressure_mean
)

print(
    df_with_missing[
        [
            "machine",
            "pressure"
        ]
    ]
)

print()


# ==================================================
# 19. SAVE MACHINE DATABASE
# ==================================================

print("TEST 19: Save Machine Database")
print()

database_file = "machine_database.csv"

df.to_csv(
    database_file,
    index=False
)

print(
    "Machine database saved as:",
    database_file
)

print()


# ==================================================
# 20. LOAD MACHINE DATABASE
# ==================================================

print("TEST 20: Load Machine Database")
print()

loaded_df = pd.read_csv(
    database_file
)

print(loaded_df)

print()


# ==================================================
# 21. CHECK LOADED DATA
# ==================================================

print("TEST 21: Verify Loaded Database")
print()

print(
    "Rows:",
    loaded_df.shape[0]
)

print(
    "Columns:",
    loaded_df.shape[1]
)

print()

print(
    "Database successfully loaded."
)

print()


# ==================================================
# 22. IDENTIFY FEATURES
# ==================================================

print("TEST 22: Identify ML Features")
print()

features = df[
    [
        "temperature",
        "pressure",
        "rpm",
        "operating_hours"
    ]
]

print("Features:")
print(features)

print()


# ==================================================
# 23. IDENTIFY TARGET
# ==================================================

print("TEST 23: Identify ML Target")
print()

target = df[
    "risk_score"
]

print("Target:")
print(target)

print()


# ==================================================
# 24. FEATURE MATRIX
# ==================================================

print("TEST 24: Feature Matrix")
print()

X = features.to_numpy()

print(X)

print()

print("Feature matrix shape:")
print(X.shape)

print()


# ==================================================
# 25. TARGET VECTOR
# ==================================================

print("TEST 25: Target Vector")
print()

y = target.to_numpy()

print(y)

print()

print("Target vector shape:")
print(y.shape)

print()


# ==================================================
# 26. FINAL MACHINE LEARNING STRUCTURE
# ==================================================

print("MACHINE LEARNING STRUCTURE")
print()

print("Features:")
print(
    [
        "temperature",
        "pressure",
        "rpm",
        "operating_hours"
    ]
)

print()

print("Target:")
print("risk_score")

print()

print("Feature matrix X:")
print(X)

print()

print("Target vector y:")
print(y)

print()

print(
    "X contains the inputs used by a model."
)

print(
    "y contains the value the model learns to predict."
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 18 COMPLETE ===")
