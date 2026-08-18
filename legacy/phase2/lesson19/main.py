# Silverwing ML
# Phase 2 - Lesson 19
# Data Cleaning and Preprocessing

import pandas as pd
import numpy as np


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 19")
print("Data Cleaning and Preprocessing")
print()


# ==================================================
# 1. CREATE INTENTIONALLY MESSY DATA
# ==================================================

data = {
    "machine": [
        "Pump",
        "Compressor",
        "Generator",
        "Turbine",
        "Boiler",
        "Pump"
    ],

    "temperature": [
        85,
        72,
        np.nan,
        91,
        98,
        85
    ],

    "pressure": [
        120,
        150,
        110,
        np.nan,
        145,
        120
    ],

    "rpm": [
        1500,
        2800,
        3200,
        2900,
        "2600",
        1500
    ],

    "operating_hours": [
        2500,
        3200,
        4500,
        3800,
        4200,
        2500
    ],

    "risk_score": [
        20,
        25,
        80,
        45,
        np.nan,
        20
    ]
}


df = pd.DataFrame(data)


# ==================================================
# 2. DISPLAY RAW DATA
# ==================================================

print("TEST 1: Raw Data")
print()

print(df)

print()


# ==================================================
# 3. CHECK DATA TYPES
# ==================================================

print("TEST 2: Data Types")
print()

print(df.dtypes)

print()


# ==================================================
# 4. CHECK MISSING VALUES
# ==================================================

print("TEST 3: Missing Values")
print()

missing_values = df.isnull().sum()

print(missing_values)

print()


# ==================================================
# 5. CALCULATE MISSING VALUE PERCENTAGE
# ==================================================

print("TEST 4: Missing Value Percentage")
print()

missing_percentage = (
        df.isnull().mean() * 100
)

print(missing_percentage)

print()


# ==================================================
# 6. CONVERT RPM TO NUMERIC
# ==================================================

print("TEST 5: Convert RPM to Numeric")
print()

df["rpm"] = pd.to_numeric(
    df["rpm"],
    errors="coerce"
)

print(df)

print()

print("Updated data types:")

print(df.dtypes)

print()


# ==================================================
# 7. FILL MISSING TEMPERATURE
# ==================================================

print("TEST 6: Fill Missing Temperature")
print()

temperature_mean = df[
    "temperature"
].mean()

df["temperature"] = df[
    "temperature"
].fillna(
    temperature_mean
)

print(df["temperature"])

print()


# ==================================================
# 8. FILL MISSING PRESSURE
# ==================================================

print("TEST 7: Fill Missing Pressure")
print()

pressure_mean = df[
    "pressure"
].mean()

df["pressure"] = df[
    "pressure"
].fillna(
    pressure_mean
)

print(df["pressure"])

print()


# ==================================================
# 9. FILL MISSING RISK SCORE
# ==================================================

print("TEST 8: Fill Missing Target")
print()

risk_mean = df[
    "risk_score"
].mean()

df["risk_score"] = df[
    "risk_score"
].fillna(
    risk_mean
)

print(df["risk_score"])

print()


# ==================================================
# 10. CHECK FOR DUPLICATES
# ==================================================

print("TEST 9: Duplicate Detection")
print()

duplicates = df.duplicated()

print(duplicates)

print()

print(
    "Number of duplicate rows:",
    duplicates.sum()
)

print()


# ==================================================
# 11. REMOVE DUPLICATES
# ==================================================

print("TEST 10: Remove Duplicates")
print()

before_count = len(df)

df = df.drop_duplicates()

after_count = len(df)

print(
    "Rows before:",
    before_count
)

print(
    "Rows after:",
    after_count
)

print()


# ==================================================
# 12. CHECK FOR REMAINING MISSING VALUES
# ==================================================

print("TEST 11: Verify Missing Values")
print()

print(
    df.isnull().sum()
)

print()


# ==================================================
# 13. VALIDATE NUMERIC RANGES
# ==================================================

print("TEST 12: Validate Temperature Range")
print()

valid_temperature = (
        (df["temperature"] >= -50)
        &
        (df["temperature"] <= 200)
)

print(
    "Valid temperatures:"
)

print(valid_temperature)

print()


# ==================================================
# 14. VALIDATE RPM
# ==================================================

print("TEST 13: Validate RPM")
print()

valid_rpm = (
        (df["rpm"] >= 0)
        &
        (df["rpm"] <= 10000)
)

print(
    "Valid RPM values:"
)

print(valid_rpm)

print()


# ==================================================
# 15. REMOVE INVALID RECORDS
# ==================================================

print("TEST 14: Remove Invalid Records")
print()

valid_rows = (
        valid_temperature
        &
        valid_rpm
)

clean_df = df[
    valid_rows
].copy()

print(clean_df)

print()


# ==================================================
# 16. DETECT EXTREME VALUES
# ==================================================

print("TEST 15: Detect Extreme Temperature")
print()

temperature_mean = clean_df[
    "temperature"
].mean()

temperature_std = clean_df[
    "temperature"
].std()

upper_limit = (
        temperature_mean
        +
        3 * temperature_std
)

lower_limit = (
        temperature_mean
        -
        3 * temperature_std
)

print(
    "Mean:",
    temperature_mean
)

print(
    "Standard deviation:",
    temperature_std
)

print(
    "Lower limit:",
    lower_limit
)

print(
    "Upper limit:",
    upper_limit
)

print()


# ==================================================
# 17. NORMALIZATION
# ==================================================

print("TEST 16: Min-Max Normalization")
print()

temperature_min = clean_df[
    "temperature"
].min()

temperature_max = clean_df[
    "temperature"
].max()

clean_df[
    "temperature_normalized"
] = (
        (
                clean_df["temperature"]
                -
                temperature_min
        )
        /
        (
                temperature_max
                -
                temperature_min
        )
)


print(
    clean_df[
        [
            "machine",
            "temperature",
            "temperature_normalized"
        ]
    ]
)

print()


# ==================================================
# 18. STANDARDIZATION
# ==================================================

print("TEST 17: Standardization")
print()

clean_df[
    "temperature_standardized"
] = (
        (
                clean_df["temperature"]
                -
                clean_df["temperature"].mean()
        )
        /
        clean_df["temperature"].std()
)


print(
    clean_df[
        [
            "machine",
            "temperature",
            "temperature_standardized"
        ]
    ]
)

print()


# ==================================================
# 19. PREPARE FEATURES
# ==================================================

print("TEST 18: Prepare Features")
print()

feature_columns = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]

X = clean_df[
    feature_columns
].copy()

print("Features:")
print(X)

print()


# ==================================================
# 20. PREPARE TARGET
# ==================================================

print("TEST 19: Prepare Target")
print()

y = clean_df[
    "risk_score"
].copy()

print("Target:")
print(y)

print()


# ==================================================
# 21. CONVERT TO NUMPY
# ==================================================

print("TEST 20: Convert to NumPy")
print()

X_array = X.to_numpy()
y_array = y.to_numpy()

print("X shape:")
print(X_array.shape)

print()

print("y shape:")
print(y_array.shape)

print()


# ==================================================
# 22. SAVE CLEAN DATA
# ==================================================

print("TEST 21: Save Clean Dataset")
print()

clean_file = "clean_machine_database.csv"

clean_df.to_csv(
    clean_file,
    index=False
)

print(
    "Clean dataset saved as:",
    clean_file
)

print()


# ==================================================
# 23. FINAL DATA QUALITY CHECK
# ==================================================

print("TEST 22: Final Data Quality Check")
print()

print("Missing values:")

print(
    clean_df.isnull().sum()
)

print()

print("Rows:", clean_df.shape[0])
print("Columns:", clean_df.shape[1])

print()


# ==================================================
# 24. MACHINE LEARNING PIPELINE
# ==================================================

print("MACHINE LEARNING PIPELINE")
print()

print("1. Raw data")
print("2. Missing-value handling")
print("3. Type conversion")
print("4. Duplicate removal")
print("5. Range validation")
print("6. Feature preprocessing")
print("7. Feature matrix X")
print("8. Target vector y")
print("9. Ready for model training")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 19 COMPLETE ===")
