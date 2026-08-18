# Silverwing ML
# Phase 1 - Lesson 06
# Analysis Module


def analyze_temperature(temperature):

    if temperature >= 100:
        return "CRITICAL"

    elif temperature >= 80:
        return "HIGH"

    else:
        return "NORMAL"


def analyze_rpm(rpm):

    if rpm > 3000:
        return "HIGH"

    elif rpm > 2500:
        return "ELEVATED"

    else:
        return "NORMAL"


def analyze_pressure(pressure):

    if pressure >= 160:
        return "CRITICAL"

    elif pressure >= 130:
        return "HIGH"

    else:
        return "NORMAL"


def calculate_risk(machine):

    temperature = machine["temperature"]
    pressure = machine["pressure"]
    rpm = machine["rpm"]

    risk_score = 0

    if temperature >= 100:
        risk_score += 40

    elif temperature >= 80:
        risk_score += 20

    if rpm > 3000:
        risk_score += 40

    elif rpm > 2500:
        risk_score += 15

    if pressure >= 160:
        risk_score += 20

    elif pressure >= 130:
        risk_score += 10

    return risk_score


def classify_risk(risk_score):

    if risk_score >= 70:
        return "CRITICAL"

    elif risk_score >= 40:
        return "HIGH"

    elif risk_score >= 20:
        return "MEDIUM"

    else:
        return "LOW"


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

    return {
        "name": machine["name"],
        "temperature_status": temperature_status,
        "rpm_status": rpm_status,
        "pressure_status": pressure_status,
        "risk_score": risk_score,
        "risk_level": risk_level
    }
