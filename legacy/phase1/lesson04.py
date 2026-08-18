# Silverwing ML
# Phase 1 - Lesson 04
# Loops, Conditions and Basic Algorithms


print("=== SILVERWING ML ===")
print("Lesson 04: Data Processing and Algorithms")
print()


# ==================================================
# 1. BASIC LOOP
# ==================================================

temperatures = [72, 85, 91, 68, 105, 79, 88]

print("TEMPERATURE READINGS")
print()

for temperature in temperatures:
    print(temperature)

print()


# ==================================================
# 2. FIND HIGH TEMPERATURES
# ==================================================

print("HIGH TEMPERATURES")
print()

for temperature in temperatures:

    if temperature >= 80:
        print(temperature)

print()


# ==================================================
# 3. COUNT HIGH TEMPERATURES
# ==================================================

high_temperature_count = 0

for temperature in temperatures:

    if temperature >= 80:
        high_temperature_count += 1


print("Number of high temperature readings:",
      high_temperature_count)

print()


# ==================================================
# 4. FIND THE HIGHEST TEMPERATURE
# ==================================================

highest_temperature = temperatures[0]

for temperature in temperatures:

    if temperature > highest_temperature:
        highest_temperature = temperature


print("Highest temperature:",
      highest_temperature)

print()


# ==================================================
# 5. FIND THE LOWEST TEMPERATURE
# ==================================================

lowest_temperature = temperatures[0]

for temperature in temperatures:

    if temperature < lowest_temperature:
        lowest_temperature = temperature


print("Lowest temperature:",
      lowest_temperature)

print()


# ==================================================
# 6. CALCULATE AVERAGE
# ==================================================

total_temperature = 0

for temperature in temperatures:
    total_temperature += temperature


average_temperature = (
        total_temperature / len(temperatures)
)


print("Average temperature:",
      average_temperature)

print()


# ==================================================
# 7. CLASSIFY TEMPERATURES
# ==================================================

print("TEMPERATURE CLASSIFICATION")
print()

for temperature in temperatures:

    if temperature >= 100:
        status = "CRITICAL"

    elif temperature >= 80:
        status = "HIGH"

    else:
        status = "NORMAL"

    print(
        "Temperature:",
        temperature,
        "->",
        status
    )

print()


# ==================================================
# 8. FILTER NORMAL TEMPERATURES
# ==================================================

normal_temperatures = []

for temperature in temperatures:

    if temperature < 80:
        normal_temperatures.append(temperature)


print("Normal temperatures:")
print(normal_temperatures)
print()


# ==================================================
# 9. FILTER CRITICAL TEMPERATURES
# ==================================================

critical_temperatures = []

for temperature in temperatures:

    if temperature >= 100:
        critical_temperatures.append(temperature)


print("Critical temperatures:")
print(critical_temperatures)
print()


# ==================================================
# 10. PROCESS MACHINE DATA
# ==================================================

machines = [
    {
        "name": "Pump",
        "temperature": 85,
        "rpm": 1500
    },

    {
        "name": "Compressor",
        "temperature": 72,
        "rpm": 2800
    },

    {
        "name": "Generator",
        "temperature": 105,
        "rpm": 3200
    },

    {
        "name": "Turbine",
        "temperature": 91,
        "rpm": 2900
    }
]


print("MACHINE ANALYSIS")
print()


for machine in machines:

    temperature = machine["temperature"]
    rpm = machine["rpm"]

    if temperature >= 100:

        status = "CRITICAL"

    elif temperature >= 80 or rpm > 3000:

        status = "WARNING"

    else:

        status = "NORMAL"


    print(
        machine["name"],
        "->",
        status
    )


print()


# ==================================================
# 11. CREATE A SIMPLE RISK SCORE
# ==================================================

print("MACHINE RISK SCORES")
print()


for machine in machines:

    temperature = machine["temperature"]
    rpm = machine["rpm"]

    risk_score = 0


    if temperature >= 100:
        risk_score += 50

    elif temperature >= 80:
        risk_score += 25


    if rpm > 3000:
        risk_score += 50

    elif rpm > 2500:
        risk_score += 20


    print(
        machine["name"],
        "Risk Score:",
        risk_score
    )


print()


# ==================================================
# 12. FIND HIGHEST RISK MACHINE
# ==================================================

highest_risk_machine = machines[0]
highest_risk_score = 0


for machine in machines:

    temperature = machine["temperature"]
    rpm = machine["rpm"]

    risk_score = 0


    if temperature >= 100:
        risk_score += 50

    elif temperature >= 80:
        risk_score += 25


    if rpm > 3000:
        risk_score += 50

    elif rpm > 2500:
        risk_score += 20


    if risk_score > highest_risk_score:

        highest_risk_score = risk_score
        highest_risk_machine = machine


print(
    "Highest risk machine:",
    highest_risk_machine["name"]
)

print(
    "Highest risk score:",
    highest_risk_score
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 04 COMPLETE ===")