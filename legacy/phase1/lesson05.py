# Silverwing ML
# Phase 1 - Lesson 05
# Functions + Data Processing


print("=== SILVERWING ML ===")
print("Lesson 05: Functions + Data Processing")
print()


# ==================================================
# 1. MACHINE DATA
# ==================================================

machines = [
    {
        "name": "Pump",
        "temperature": 85,
        "pressure": 120,
        "rpm": 1500,
        "operating_hours": 2500
    },

    {
        "name": "Compressor",
        "temperature": 72,
        "pressure": 150,
        "rpm": 2800,
        "operating_hours": 3200
    },

    {
        "name": "Generator",
        "temperature": 105,
        "pressure": 110,
        "rpm": 3200,
        "operating_hours": 4500
    },

    {
        "name": "Turbine",
        "temperature": 91,
        "pressure": 135,
        "rpm": 2900,
        "operating_hours": 3800
    }
]


# ==================================================
# 2. TEMPERATURE ANALYSIS
# ==================================================

def analyze_temperature(temperature):

    if temperature >= 100:
        return "CRITICAL"

    elif temperature >= 80:
        return "HIGH"

    else:
        return "NORMAL"


# ==================================================
# 3. RPM ANALYSIS
# ==================================================

def analyze_rpm(rpm):

    if rpm > 3000:
        return "HIGH"

    elif rpm > 2500:
        return "ELEVATED"

    else:
        return "NORMAL"


# ==================================================
# 4. PRESSURE ANALYSIS
# ==================================================

def analyze_pressure(pressure):

    if pressure >= 160:
        return "CRITICAL"

    elif pressure >= 130:
        return "HIGH"

    else:
        return "NORMAL"


# ==================================================
# 5. CALCULATE MACHINE RISK
# ==================================================

def calculate_risk(machine):

    temperature = machine["temperature"]
    pressure = machine["pressure"]
    rpm = machine["rpm"]

    risk_score = 0


    # Temperature contribution

    if temperature >= 100:
        risk_score += 40

    elif temperature >= 80:
        risk_score += 20


    # RPM contribution

    if rpm > 3000:
        risk_score += 40

    elif rpm > 2500:
        risk_score += 15


    # Pressure contribution

    if pressure >= 160:
        risk_score += 20

    elif pressure >= 130:
        risk_score += 10


    return risk_score


# ==================================================
# 6. CLASSIFY MACHINE RISK
# ==================================================

def classify_risk(risk_score):

    if risk_score >= 70:
        return "CRITICAL"

    elif risk_score >= 40:
        return "HIGH"

    elif risk_score >= 20:
        return "MEDIUM"

    else:
        return "LOW"


# ==================================================
# 7. ANALYZE ONE MACHINE
# ==================================================

def analyze_machine(machine):

    temperature_status = analyze_temperature(
        machine["temperature"]
    )

    rpm_status = analyze_rpm(
        machine["rpm"]
    )

    pressure_status = analyze_pressure(
        machine["pressure"]
    )

    risk_score = calculate_risk(machine)

    risk_level = classify_risk(
        risk_score
    )


    result = {
        "name": machine["name"],
        "temperature_status": temperature_status,
        "rpm_status": rpm_status,
        "pressure_status": pressure_status,
        "risk_score": risk_score,
        "risk_level": risk_level
    }


    return result


# ==================================================
# 8. ANALYZE ALL MACHINES
# ==================================================

analysis_results = []


for machine in machines:

    result = analyze_machine(machine)

    analysis_results.append(result)


# ==================================================
# 9. DISPLAY RESULTS
# ==================================================

print("MACHINE ANALYSIS")
print()


for result in analysis_results:

    print("Machine:", result["name"])

    print(
        "Temperature:",
        result["temperature_status"]
    )

    print(
        "RPM:",
        result["rpm_status"]
    )

    print(
        "Pressure:",
        result["pressure_status"]
    )

    print(
        "Risk Score:",
        result["risk_score"]
    )

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print("-" * 40)


# ==================================================
# 10. FIND HIGHEST-RISK MACHINE
# ==================================================

highest_risk = analysis_results[0]


for result in analysis_results:

    if result["risk_score"] > highest_risk["risk_score"]:

        highest_risk = result


print()

print("HIGHEST-RISK MACHINE")
print()

print(
    "Machine:",
    highest_risk["name"]
)

print(
    "Risk Score:",
    highest_risk["risk_score"]
)

print(
    "Risk Level:",
    highest_risk["risk_level"]
)


# ==================================================
# 11. CALCULATE AVERAGE TEMPERATURE
# ==================================================

total_temperature = 0


for machine in machines:

    total_temperature += machine["temperature"]


average_temperature = (
        total_temperature / len(machines)
)


print()

print(
    "Average Machine Temperature:",
    average_temperature
)


# ==================================================
# 12. COUNT CRITICAL MACHINES
# ==================================================

critical_count = 0


for result in analysis_results:

    if result["risk_level"] == "CRITICAL":

        critical_count += 1


print(
    "Critical Machines:",
    critical_count
)


# ==================================================
# LESSON COMPLETE
# ==================================================

print()
print("=== LESSON 05 COMPLETE ===")