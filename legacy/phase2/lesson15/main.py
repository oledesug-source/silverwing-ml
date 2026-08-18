# Silverwing ML
# Phase 2 - Lesson 15
# Probability Foundations


import random
import math


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 15")
print("Probability Foundations")
print()


# ==================================================
# 1. BASIC PROBABILITY
# ==================================================

print("TEST 1: Basic Probability")
print()

# A standard six-sided die has six possible outcomes.

possible_outcomes = 6

favorable_outcomes = 1

probability_six = (
        favorable_outcomes
        /
        possible_outcomes
)


print(
    "Probability of rolling a 6:",
    probability_six
)

print()


# ==================================================
# 2. PROBABILITY AS A PERCENTAGE
# ==================================================

print("TEST 2: Probability Percentage")
print()

percentage = probability_six * 100

print(
    "Probability percentage:",
    percentage,
    "%"
)

print()


# ==================================================
# 3. COIN FLIP
# ==================================================

print("TEST 3: Coin Probability")
print()

heads_probability = 1 / 2
tails_probability = 1 / 2

print(
    "Probability of heads:",
    heads_probability
)

print(
    "Probability of tails:",
    tails_probability
)

print()


# ==================================================
# 4. SUM OF PROBABILITIES
# ==================================================

total_probability = (
        heads_probability
        +
        tails_probability
)


print(
    "Total probability:",
    total_probability
)

print()


# ==================================================
# 5. RANDOM EXPERIMENT
# ==================================================

print("TEST 4: Random Experiment")
print()

random_number = random.randint(1, 6)

print(
    "Random die result:",
    random_number
)

print()


# ==================================================
# 6. SIMULATE MANY DIE ROLLS
# ==================================================

print("TEST 5: Simulation")
print()

number_of_rolls = 10000

six_count = 0


for _ in range(number_of_rolls):

    result = random.randint(1, 6)

    if result == 6:
        six_count += 1


experimental_probability = (
        six_count
        /
        number_of_rolls
)


print(
    "Number of rolls:",
    number_of_rolls
)

print(
    "Number of sixes:",
    six_count
)

print(
    "Experimental probability:",
    experimental_probability
)

print(
    "Theoretical probability:",
    1 / 6
)

print()


# ==================================================
# 7. MACHINE FAILURE PROBABILITY
# ==================================================

print("TEST 6: Machine Failure Probability")
print()


machines_tested = 1000

machines_failed = 80


failure_probability = (
        machines_failed
        /
        machines_tested
)


print(
    "Machines tested:",
    machines_tested
)

print(
    "Machines failed:",
    machines_failed
)

print(
    "Estimated failure probability:",
    failure_probability
)

print(
    "Estimated failure percentage:",
    failure_probability * 100,
    "%"
)

print()


# ==================================================
# 8. COMPLEMENT PROBABILITY
# ==================================================

print("TEST 7: Complement Probability")
print()


failure_probability = 0.08

normal_probability = (
        1 - failure_probability
)


print(
    "Failure probability:",
    failure_probability
)

print(
    "Normal operation probability:",
    normal_probability
)

print(
    "Total:",
    failure_probability + normal_probability
)

print()


# ==================================================
# 9. CONDITIONAL PROBABILITY IDEA
# ==================================================

print("TEST 8: Conditional Probability")
print()


total_machines = 1000

hot_machines = 200

failed_hot_machines = 50


probability_failure_given_hot = (
        failed_hot_machines
        /
        hot_machines
)


print(
    "Hot machines:",
    hot_machines
)

print(
    "Failed hot machines:",
    failed_hot_machines
)

print(
    "Probability of failure given high temperature:",
    probability_failure_given_hot
)

print(
    "Percentage:",
    probability_failure_given_hot * 100,
    "%"
)

print()


# ==================================================
# 10. MULTIPLE OUTCOMES
# ==================================================

print("TEST 9: Multiple Model Outcomes")
print()


normal_probability = 0.10
warning_probability = 0.25
failure_probability = 0.65


total_probability = (
        normal_probability
        +
        warning_probability
        +
        failure_probability
)


print(
    "Normal:",
    normal_probability
)

print(
    "Warning:",
    warning_probability
)

print(
    "Failure:",
    failure_probability
)

print(
    "Total:",
    total_probability
)

print()


# ==================================================
# 11. CHOOSE MOST PROBABLE OUTCOME
# ==================================================

print("TEST 10: Most Probable Outcome")
print()


probabilities = {
    "NORMAL": 0.10,
    "WARNING": 0.25,
    "FAILURE": 0.65
}


most_likely = max(
    probabilities,
    key=probabilities.get
)


print(
    "Most probable outcome:",
    most_likely
)

print(
    "Probability:",
    probabilities[most_likely]
)

print()


# ==================================================
# 12. RANDOM SAMPLING
# ==================================================

print("TEST 11: Random Sampling")
print()


population = [
    "NORMAL",
    "NORMAL",
    "NORMAL",
    "WARNING",
    "WARNING",
    "FAILURE"
]


sample = random.choice(population)


print(
    "Randomly selected observation:",
    sample
)

print()


# ==================================================
# 13. MEAN AND VARIABILITY
# ==================================================

print("TEST 12: Mean and Variability")
print()


values = [
    10,
    12,
    14,
    16,
    18
]


mean = sum(values) / len(values)


variance = sum(
    (value - mean) ** 2
    for value in values
) / len(values)


standard_deviation = math.sqrt(
    variance
)


print(
    "Values:",
    values
)

print(
    "Mean:",
    mean
)

print(
    "Variance:",
    variance
)

print(
    "Standard deviation:",
    standard_deviation
)

print()


# ==================================================
# 14. MACHINE SENSOR DATA
# ==================================================

print("TEST 13: Sensor Probability")
print()


sensor_readings = [
    70,
    72,
    75,
    80,
    82,
    85,
    90,
    95,
    100,
    105
]


high_readings = 0


for reading in sensor_readings:

    if reading >= 80:
        high_readings += 1


probability_high = (
        high_readings
        /
        len(sensor_readings)
)


print(
    "Sensor readings:",
    sensor_readings
)

print(
    "High readings:",
    high_readings
)

print(
    "Probability of high reading:",
    probability_high
)

print()


# ==================================================
# 15. ML CONNECTION
# ==================================================

print("TEST 14: Machine Learning Connection")
print()

print("Machine learning deals with uncertainty.")

print()

print("A model may produce:")

print(
    "NORMAL   = 0.10"
)

print(
    "WARNING  = 0.25"
)

print(
    "FAILURE  = 0.65"
)

print()

print(
    "The model predicts the outcome with "
    "the highest estimated probability."
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 15 COMPLETE ===")