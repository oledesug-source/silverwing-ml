# Silverwing ML
# Phase 2 - Lesson 16
# Descriptive Statistics


import numpy as np


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 16")
print("Descriptive Statistics")
print()


# ==================================================
# 1. DATASET
# ==================================================

temperatures = np.array([
    70,
    72,
    75,
    78,
    80,
    82,
    85,
    88,
    90,
    95
])


print("TEST 1: Dataset")
print()

print("Temperature data:")
print(temperatures)

print()

print("Number of observations:")
print(len(temperatures))

print()


# ==================================================
# 2. MEAN
# ==================================================

print("TEST 2: Mean")
print()

mean_temperature = np.mean(temperatures)

print("Mean temperature:")
print(mean_temperature)

print()


# ==================================================
# 3. MEDIAN
# ==================================================

print("TEST 3: Median")
print()

median_temperature = np.median(temperatures)

print("Median temperature:")
print(median_temperature)

print()


# ==================================================
# 4. MINIMUM AND MAXIMUM
# ==================================================

print("TEST 4: Minimum and Maximum")
print()

minimum_temperature = np.min(temperatures)
maximum_temperature = np.max(temperatures)

print("Minimum:", minimum_temperature)
print("Maximum:", maximum_temperature)

print()


# ==================================================
# 5. RANGE
# ==================================================

print("TEST 5: Range")
print()

temperature_range = (
        maximum_temperature
        -
        minimum_temperature
)

print("Range:")
print(temperature_range)

print()


# ==================================================
# 6. VARIANCE
# ==================================================

print("TEST 6: Variance")
print()

temperature_variance = np.var(
    temperatures
)

print("Variance:")
print(temperature_variance)

print()


# ==================================================
# 7. STANDARD DEVIATION
# ==================================================

print("TEST 7: Standard Deviation")
print()

temperature_std = np.std(
    temperatures
)

print("Standard deviation:")
print(temperature_std)

print()


# ==================================================
# 8. PERCENTILES
# ==================================================

print("TEST 8: Percentiles")
print()

percentile_25 = np.percentile(
    temperatures,
    25
)

percentile_50 = np.percentile(
    temperatures,
    50
)

percentile_75 = np.percentile(
    temperatures,
    75
)

print("25th percentile:", percentile_25)
print("50th percentile:", percentile_50)
print("75th percentile:", percentile_75)

print()


# ==================================================
# 9. MEDIAN AND 50TH PERCENTILE
# ==================================================

print("TEST 9: Median Verification")
print()

print("Median:", median_temperature)
print("50th percentile:", percentile_50)

print()


# ==================================================
# 10. DETECT HIGH VALUES
# ==================================================

print("TEST 10: High Temperature Detection")
print()

high_temperatures = temperatures[
    temperatures >= 85
    ]

print("Temperatures >= 85:")
print(high_temperatures)

print()

print(
    "Number of high temperatures:",
    len(high_temperatures)
)

print()


# ==================================================
# 11. DETECT LOW VALUES
# ==================================================

print("TEST 11: Low Temperature Detection")
print()

low_temperatures = temperatures[
    temperatures < 80
    ]

print("Temperatures < 80:")
print(low_temperatures)

print()

print(
    "Number of low temperatures:",
    len(low_temperatures)
)

print()


# ==================================================
# 12. MACHINE SENSOR DATA
# ==================================================

print("TEST 12: Machine Sensor Dataset")
print()


machine_temperatures = np.array([
    72,
    75,
    78,
    80,
    84,
    87,
    91,
    95,
    102,
    110
])


machine_pressures = np.array([
    100,
    105,
    108,
    110,
    115,
    118,
    120,
    125,
    135,
    145
])


machine_rpm = np.array([
    1400,
    1500,
    1550,
    1600,
    1700,
    1800,
    2000,
    2200,
    2700,
    3200
])


print("Temperature:")
print(machine_temperatures)

print()

print("Pressure:")
print(machine_pressures)

print()

print("RPM:")
print(machine_rpm)

print()


# ==================================================
# 13. SUMMARIZE SENSOR DATA
# ==================================================

def summarize_dataset(name, data):

    print(name)
    print("-" * 40)

    print("Count:", len(data))

    print("Mean:", np.mean(data))

    print("Median:", np.median(data))

    print("Minimum:", np.min(data))

    print("Maximum:", np.max(data))

    print("Range:", np.ptp(data))

    print("Variance:", np.var(data))

    print("Standard deviation:", np.std(data))

    print()


print("TEST 13: Dataset Summaries")
print()

summarize_dataset(
    "TEMPERATURE",
    machine_temperatures
)

summarize_dataset(
    "PRESSURE",
    machine_pressures
)

summarize_dataset(
    "RPM",
    machine_rpm
)


# ==================================================
# 14. OUTLIER EXAMPLE
# ==================================================

print("TEST 14: Possible Outlier")
print()

normal_data = np.array([
    70,
    72,
    74,
    75,
    76,
    78,
    80
])


data_with_outlier = np.array([
    70,
    72,
    74,
    75,
    76,
    78,
    200
])


print("Normal dataset:")
print(normal_data)

print()

print("Dataset containing unusual value:")
print(data_with_outlier)

print()

print(
    "Normal mean:",
    np.mean(normal_data)
)

print(
    "Mean with unusual value:",
    np.mean(data_with_outlier)
)

print()


# ==================================================
# 15. MEAN COMPARISON
# ==================================================

print("TEST 15: Effect of an Outlier")
print()

print(
    "The unusual value changes the mean significantly."
)

print(
    "This is why data analysis is important before "
    "training a machine-learning model."
)

print()


# ==================================================
# 16. SIMPLE STATISTICAL REPORT
# ==================================================

print("TEST 16: Statistical Report")
print()

average_temperature = np.mean(
    machine_temperatures
)

median_temperature = np.median(
    machine_temperatures
)

temperature_std = np.std(
    machine_temperatures
)


print("Average temperature:", average_temperature)
print("Median temperature:", median_temperature)
print("Temperature variation:", temperature_std)

print()


# ==================================================
# 17. ML CONNECTION
# ==================================================

print("TEST 17: Machine Learning Connection")
print()

print("Before training an ML model, we inspect the data.")

print()

print("We ask:")

print("1. What is the average?")
print("2. How spread out is the data?")
print("3. What values are unusual?")
print("4. What is the minimum?")
print("5. What is the maximum?")
print("6. Are there possible outliers?")

print()

print(
    "Statistics helps us understand the dataset "
    "before machine learning begins."
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 16 COMPLETE ===")
