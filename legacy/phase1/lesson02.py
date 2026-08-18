# Silverwing ML
# Phase 1 - Lesson 02
# Python Functions


def check_temperature(temperature):
    if temperature >= 100:
        return "CRITICAL"
    elif temperature >= 80:
        return "HIGH"
    else:
        return "NORMAL"


def check_rpm(rpm):
    if rpm > 3000:
        return "HIGH RPM"
    else:
        return "RPM NORMAL"


def calculate_temperature_pressure(temperature, pressure):
    return temperature * pressure


def machine_status(temperature, rpm):
    temperature_status = check_temperature(temperature)
    rpm_status = check_rpm(rpm)

    if temperature_status == "CRITICAL":
        return "CRITICAL"

    if temperature_status == "HIGH" or rpm_status == "HIGH RPM":
        return "WARNING"

    return "NORMAL"


# Machine information
machine = "Pump"
temperature = 85
pressure = 120
rpm = 1500
operating_hours = 2500


# Use our functions
temperature_status = check_temperature(temperature)
rpm_status = check_rpm(rpm)
temperature_pressure = calculate_temperature_pressure(
    temperature,
    pressure
)
overall_status = machine_status(temperature, rpm)


# Display information
print("=== SILVERWING ML ===")
print("Lesson 02: Python Functions")
print()

print("Machine:", machine)
print("Temperature:", temperature)
print("Pressure:", pressure)
print("RPM:", rpm)
print("Operating Hours:", operating_hours)
print()

print("Temperature Status:", temperature_status)
print("RPM Status:", rpm_status)
print("Temperature × Pressure:", temperature_pressure)
print("Overall Machine Status:", overall_status)

print()
print("=== LESSON 02 COMPLETE ===")