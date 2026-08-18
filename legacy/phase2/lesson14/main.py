# Silverwing ML
# Phase 2 - Lesson 14
# Vectors and Matrices
# Foundation for Neural Networks and LLMs


import numpy as np


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 14")
print("Vectors and Matrices")
print()


# ==================================================
# 1. SCALAR
# ==================================================

print("TEST 1: Scalar")
print()

temperature = 85

print("Temperature:", temperature)
print("Type:", type(temperature))

print()


# ==================================================
# 2. VECTOR
# ==================================================

print("TEST 2: Vector")
print()

temperatures = np.array([
    70,
    75,
    80,
    85,
    90
])

print("Temperature vector:")
print(temperatures)

print()

print("Number of elements:")
print(len(temperatures))

print()


# ==================================================
# 3. VECTOR INDEXING
# ==================================================

print("TEST 3: Vector Indexing")
print()

print("First temperature:", temperatures[0])
print("Second temperature:", temperatures[1])
print("Last temperature:", temperatures[-1])

print()


# ==================================================
# 4. VECTOR OPERATIONS
# ==================================================

print("TEST 4: Vector Operations")
print()

temperature_adjustment = np.array([
    2,
    2,
    2,
    2,
    2
])

new_temperatures = (
        temperatures
        +
        temperature_adjustment
)


print("Original:")
print(temperatures)

print()

print("Adjustment:")
print(temperature_adjustment)

print()

print("New temperatures:")
print(new_temperatures)

print()


# ==================================================
# 5. VECTOR MULTIPLICATION
# ==================================================

print("TEST 5: Vector Multiplication")
print()

scaled_temperatures = temperatures * 2

print("Original:")
print(temperatures)

print()

print("Scaled:")
print(scaled_temperatures)

print()


# ==================================================
# 6. DOT PRODUCT
# ==================================================

print("TEST 6: Dot Product")
print()

weights = np.array([
    0.2,
    0.3,
    0.1
])

sensor_values = np.array([
    10,
    20,
    30
])


dot_product = np.dot(
    weights,
    sensor_values
)


print("Weights:")
print(weights)

print()

print("Sensor values:")
print(sensor_values)

print()

print("Dot product:")
print(dot_product)

print()


# ==================================================
# 7. MACHINE FEATURES
# ==================================================

print("TEST 7: Machine Feature Vector")
print()

machine_features = np.array([
    85,     # temperature
    120,    # pressure
    1500,   # RPM
    2500    # operating hours
])


print("Machine feature vector:")
print(machine_features)

print()

print("Temperature:", machine_features[0])
print("Pressure:", machine_features[1])
print("RPM:", machine_features[2])
print("Operating hours:", machine_features[3])

print()


# ==================================================
# 8. MATRIX
# ==================================================

print("TEST 8: Matrix")
print()

machine_matrix = np.array([
    [85, 120, 1500],
    [72, 150, 2800],
    [105, 110, 3200],
    [91, 135, 2900]
])


print("Machine matrix:")
print(machine_matrix)

print()

print("Rows:", machine_matrix.shape[0])
print("Columns:", machine_matrix.shape[1])

print()


# ==================================================
# 9. MATRIX ROWS
# ==================================================

print("TEST 9: Matrix Rows")
print()

print("Pump:")
print(machine_matrix[0])

print()

print("Compressor:")
print(machine_matrix[1])

print()

print("Generator:")
print(machine_matrix[2])

print()

print("Turbine:")
print(machine_matrix[3])

print()


# ==================================================
# 10. MATRIX COLUMNS
# ==================================================

print("TEST 10: Matrix Columns")
print()

temperature_column = machine_matrix[:, 0]
pressure_column = machine_matrix[:, 1]
rpm_column = machine_matrix[:, 2]


print("Temperatures:")
print(temperature_column)

print()

print("Pressure values:")
print(pressure_column)

print()

print("RPM values:")
print(rpm_column)

print()


# ==================================================
# 11. MATRIX OPERATIONS
# ==================================================

print("TEST 11: Matrix Operations")
print()

adjustment_matrix = np.array([
    [1, 2, 10],
    [1, 2, 10],
    [1, 2, 10],
    [1, 2, 10]
])


updated_matrix = (
        machine_matrix
        +
        adjustment_matrix
)


print("Original matrix:")
print(machine_matrix)

print()

print("Adjustment:")
print(adjustment_matrix)

print()

print("Updated matrix:")
print(updated_matrix)

print()


# ==================================================
# 12. MATRIX MULTIPLICATION
# ==================================================

print("TEST 12: Matrix Multiplication")
print()

features = np.array([
    [1, 2],
    [3, 4]
])

weights = np.array([
    [0.5],
    [0.25]
])


result = features @ weights


print("Features:")
print(features)

print()

print("Weights:")
print(weights)

print()

print("Result:")
print(result)

print()


# ==================================================
# 13. SIMPLE NEURAL-NETWORK IDEA
# ==================================================

print("TEST 13: Neural Network Connection")
print()

input_features = np.array([
    10,
    20,
    30
])


weights = np.array([
    0.2,
    0.3,
    0.5
])


bias = 2


weighted_sum = (
        np.dot(
            input_features,
            weights
        )
        +
        bias
)


print("Input features:")
print(input_features)

print()

print("Weights:")
print(weights)

print()

print("Bias:")
print(bias)

print()

print("Weighted sum:")
print(weighted_sum)

print()


# ==================================================
# 14. SIMPLE ACTIVATION
# ==================================================

print("TEST 14: Simple Activation")
print()


def relu(value):

    if value > 0:
        return value

    return 0


activated_value = relu(weighted_sum)


print("Weighted sum:")
print(weighted_sum)

print()

print("ReLU output:")
print(activated_value)

print()


# ==================================================
# 15. MACHINE LEARNING CONNECTION
# ==================================================

print("TEST 15: ML Connection")
print()

print("A machine can be represented as a vector:")
print()

print("[temperature, pressure, RPM, operating_hours]")

print()

print("Many machines become a matrix:")

print()

print("[")
print(" machine 1 features")
print(" machine 2 features")
print(" machine 3 features")
print(" ...")
print("]")

print()

print("A neural network performs mathematical")
print("operations on these values.")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 14 COMPLETE ===")
