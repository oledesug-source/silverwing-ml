# Silverwing ML
# Phase 2 - Lesson 13
# Mathematical Foundations for Machine Learning
# Variables, Functions, Coordinates, Slope and Linear Relationships


import numpy as np


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 13")
print("Mathematical Foundations")
print()


# ==================================================
# 1. MATHEMATICAL VARIABLES
# ==================================================

print("TEST 1: Mathematical Variables")
print()

x = 10
y = 5

print("x =", x)
print("y =", y)

print("x + y =", x + y)
print("x - y =", x - y)
print("x * y =", x * y)
print("x / y =", x / y)

print()


# ==================================================
# 2. A MATHEMATICAL FUNCTION
# ==================================================

print("TEST 2: Mathematical Function")
print()


def linear_function(x):
    return 2 * x + 3


print("f(0) =", linear_function(0))
print("f(1) =", linear_function(1))
print("f(2) =", linear_function(2))
print("f(5) =", linear_function(5))
print("f(10) =", linear_function(10))

print()


# ==================================================
# 3. INPUT AND OUTPUT
# ==================================================

print("TEST 3: Function Inputs and Outputs")
print()


input_values = [0, 1, 2, 3, 4, 5]

for value in input_values:

    output = linear_function(value)

    print(
        "x =",
        value,
        "-> f(x) =",
        output
    )

print()


# ==================================================
# 4. COORDINATE PAIRS
# ==================================================

print("TEST 4: Coordinate Pairs")
print()


coordinates = [
    (0, 3),
    (1, 5),
    (2, 7),
    (3, 9),
    (4, 11)
]


for x_value, y_value in coordinates:

    print(
        "(",
        x_value,
        ",",
        y_value,
        ")"
    )

print()


# ==================================================
# 5. SLOPE
# ==================================================

print("TEST 5: Slope")
print()


x1 = 1
y1 = 5

x2 = 4
y2 = 11


slope = (y2 - y1) / (x2 - x1)


print("First point:")
print("(", x1, ",", y1, ")")

print()

print("Second point:")
print("(", x2, ",", y2, ")")

print()

print("Slope:", slope)

print()


# ==================================================
# 6. INTERCEPT
# ==================================================

print("TEST 6: Intercept")
print()


# Our function is:
#
# y = 2x + 3
#
# Therefore:
# slope = 2
# intercept = 3


m = 2
b = 3


print("Slope:", m)
print("Intercept:", b)

print()


# ==================================================
# 7. LINEAR EQUATION
# ==================================================

print("TEST 7: Linear Equation")
print()


def calculate_y(x, m, b):

    return m * x + b


for x_value in range(6):

    y_value = calculate_y(
        x_value,
        m,
        b
    )

    print(
        "x =",
        x_value,
        "-> y =",
        y_value
    )

print()


# ==================================================
# 8. NUMPY ARRAY
# ==================================================

print("TEST 8: NumPy Arrays")
print()


x_values = np.array([
    0,
    1,
    2,
    3,
    4,
    5
])


y_values = 2 * x_values + 3


print("X values:")
print(x_values)

print()

print("Y values:")
print(y_values)

print()


# ==================================================
# 9. VECTOR OPERATIONS
# ==================================================

print("TEST 9: Vector Operations")
print()


temperatures = np.array([
    70,
    75,
    80,
    85,
    90
])


print("Temperatures:")
print(temperatures)

print()

print("Temperatures + 5:")
print(temperatures + 5)

print()

print("Temperatures × 2:")
print(temperatures * 2)

print()


# ==================================================
# 10. MACHINE DATA AS X AND Y
# ==================================================

print("TEST 10: Machine Data")
print()


operating_hours = np.array([
    500,
    1000,
    1500,
    2000,
    2500
])


temperature = np.array([
    65,
    70,
    75,
    82,
    90
])


print("Operating hours:")
print(operating_hours)

print()

print("Temperature:")
print(temperature)

print()


# ==================================================
# 11. ESTIMATE A LINEAR RELATIONSHIP
# ==================================================

print("TEST 11: Simple Relationship")
print()


x1 = operating_hours[0]
y1 = temperature[0]

x2 = operating_hours[-1]
y2 = temperature[-1]


relationship_slope = (
        (y2 - y1)
        /
        (x2 - x1)
)


print(
    "Estimated temperature change per hour:",
    relationship_slope
)

print()


# ==================================================
# 12. PREDICT USING THE RELATIONSHIP
# ==================================================

print("TEST 12: Simple Prediction")
print()


prediction_hours = 3000


predicted_temperature = (
        y1
        +
        relationship_slope
        *
        (prediction_hours - x1)
)


print(
    "Operating hours:",
    prediction_hours
)

print(
    "Estimated temperature:",
    predicted_temperature
)

print()


# ==================================================
# 13. WHY THIS MATTERS FOR ML
# ==================================================

print("TEST 13: ML Connection")
print()


print("Machine learning learns relationships between variables.")

print()

print("Example:")
print("Operating Hours -> Temperature")

print()

print("Input feature:")
print("Operating hours")

print()

print("Target:")
print("Temperature")

print()

print("The goal is to learn a function that maps:")
print("input -> output")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 13 COMPLETE ===")
