# Silverwing ML
# Phase 2 - Lesson 17
# Data Visualization with Matplotlib

import numpy as np
import matplotlib.pyplot as plt


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 17")
print("Data Visualization")
print()


# ==================================================
# 1. MACHINE DATA
# ==================================================

operating_hours = np.array([
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000
])

temperature = np.array([
    65,
    68,
    72,
    76,
    82,
    87,
    93,
    101
])

pressure = np.array([
    100,
    105,
    108,
    112,
    118,
    122,
    128,
    138
])

rpm = np.array([
    1400,
    1500,
    1600,
    1750,
    1900,
    2200,
    2600,
    3100
])


# ==================================================
# 2. PRINT DATA
# ==================================================

print("MACHINE DATA")
print()

print("Operating hours:")
print(operating_hours)

print()

print("Temperature:")
print(temperature)

print()

print("Pressure:")
print(pressure)

print()

print("RPM:")
print(rpm)

print()


# ==================================================
# 3. BASIC LINE GRAPH
# ==================================================

print("TEST 1: Line Graph")
print()

plt.figure(figsize=(8, 5))

plt.plot(
    operating_hours,
    temperature,
    marker="o"
)

plt.title(
    "Operating Hours vs Temperature"
)

plt.xlabel(
    "Operating Hours"
)

plt.ylabel(
    "Temperature"
)

plt.grid(True)

plt.tight_layout()

plt.show()

print(
    "Line graph displayed."
)

print()


# ==================================================
# 4. SCATTER PLOT
# ==================================================

print("TEST 2: Scatter Plot")
print()

plt.figure(figsize=(8, 5))

plt.scatter(
    operating_hours,
    temperature
)

plt.title(
    "Operating Hours vs Temperature"
)

plt.xlabel(
    "Operating Hours"
)

plt.ylabel(
    "Temperature"
)

plt.grid(True)

plt.tight_layout()

plt.show()

print(
    "Scatter plot displayed."
)

print()


# ==================================================
# 5. PRESSURE VS TEMPERATURE
# ==================================================

print("TEST 3: Pressure vs Temperature")
print()

plt.figure(figsize=(8, 5))

plt.scatter(
    pressure,
    temperature
)

plt.title(
    "Pressure vs Temperature"
)

plt.xlabel(
    "Pressure"
)

plt.ylabel(
    "Temperature"
)

plt.grid(True)

plt.tight_layout()

plt.show()

print()


# ==================================================
# 6. RPM VS TEMPERATURE
# ==================================================

print("TEST 4: RPM vs Temperature")
print()

plt.figure(figsize=(8, 5))

plt.scatter(
    rpm,
    temperature
)

plt.title(
    "RPM vs Temperature"
)

plt.xlabel(
    "RPM"
)

plt.ylabel(
    "Temperature"
)

plt.grid(True)

plt.tight_layout()

plt.show()

print()


# ==================================================
# 7. HISTOGRAM
# ==================================================

print("TEST 5: Temperature Distribution")
print()

plt.figure(figsize=(8, 5))

plt.hist(
    temperature,
    bins=5
)

plt.title(
    "Temperature Distribution"
)

plt.xlabel(
    "Temperature"
)

plt.ylabel(
    "Frequency"
)

plt.grid(True)

plt.tight_layout()

plt.show()

print()


# ==================================================
# 8. BAR CHART
# ==================================================

print("TEST 6: Machine Temperatures")
print()

machine_names = [
    "Pump",
    "Compressor",
    "Generator",
    "Turbine",
    "Boiler",
    "Motor",
    "Fan",
    "Press"
]

plt.figure(figsize=(10, 5))

plt.bar(
    machine_names,
    temperature
)

plt.title(
    "Machine Temperatures"
)

plt.xlabel(
    "Machine"
)

plt.ylabel(
    "Temperature"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()

print()


# ==================================================
# 9. MULTIPLE LINES
# ==================================================

print("TEST 7: Multiple Sensor Trends")
print()

plt.figure(figsize=(9, 5))

plt.plot(
    operating_hours,
    temperature,
    marker="o",
    label="Temperature"
)

plt.plot(
    operating_hours,
    pressure,
    marker="s",
    label="Pressure"
)

plt.title(
    "Sensor Trends vs Operating Hours"
)

plt.xlabel(
    "Operating Hours"
)

plt.ylabel(
    "Sensor Value"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

print()


# ==================================================
# 10. IDENTIFY THE HIGHEST TEMPERATURE
# ==================================================

highest_temperature_index = np.argmax(
    temperature
)

highest_temperature = temperature[
    highest_temperature_index
]

highest_temperature_hours = operating_hours[
    highest_temperature_index
]


print("HIGHEST TEMPERATURE")
print()

print(
    "Temperature:",
    highest_temperature
)

print(
    "Operating hours:",
    highest_temperature_hours
)

print()


# ==================================================
# 11. IDENTIFY CRITICAL READINGS
# ==================================================

critical_mask = temperature >= 100

critical_temperatures = temperature[
    critical_mask
]

critical_hours = operating_hours[
    critical_mask
]


print("CRITICAL READINGS")
print()

print(
    "Critical temperatures:",
    critical_temperatures
)

print(
    "Corresponding operating hours:",
    critical_hours
)

print()


# ==================================================
# 12. SIMPLE CORRELATION
# ==================================================

correlation_matrix = np.corrcoef(
    operating_hours,
    temperature
)

correlation = correlation_matrix[
    0,
    1
]


print("CORRELATION")
print()

print(
    "Operating hours / temperature correlation:",
    correlation
)

print()


# ==================================================
# 13. ML CONNECTION
# ==================================================

print("MACHINE LEARNING CONNECTION")
print()

print(
    "Visualization helps us discover patterns "
    "before training a model."
)

print()

print(
    "For example, if temperature increases "
    "as operating hours increase, the two "
    "variables may have a relationship."
)

print()

print(
    "A machine-learning model can later learn "
    "such relationships from historical data."
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 17 COMPLETE ===")
