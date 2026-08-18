# Silverwing ML
# Phase 1 - Lesson 07
# Error Handling and Defensive Programming


print("=== SILVERWING ML ===")
print("Lesson 07: Error Handling")
print()


# ==================================================
# 1. BASIC ERROR HANDLING
# ==================================================

print("TEST 1: Division Error")
print()

try:
    number = 10
    result = number / 0
    print("Result:", result)

except ZeroDivisionError:
    print("Error: Cannot divide by zero.")

print()


# ==================================================
# 2. CONVERT VALID DATA
# ==================================================

print("TEST 2: Valid Numeric Data")
print()

try:
    temperature = "85"
    temperature = float(temperature)

    print("Temperature:", temperature)

except ValueError:
    print("Error: Temperature must be a number.")

print()


# ==================================================
# 3. HANDLE INVALID DATA
# ==================================================

print("TEST 3: Invalid Numeric Data")
print()

try:
    temperature = "unknown"
    temperature = float(temperature)

    print("Temperature:", temperature)

except ValueError:
    print("Error: Invalid temperature data.")

print()


# ==================================================
# 4. VALIDATE TEMPERATURE
# ==================================================

def validate_temperature(value):
    """
    Convert a value to a number.

    Returns:
        float: Valid temperature
        None: If the value is invalid
    """

    try:
        return float(value)

    except (ValueError, TypeError):
        return None


valid_temperature = validate_temperature("85")

print("Validated temperature:", valid_temperature)

invalid_temperature = validate_temperature("unknown")

print("Invalid temperature:", invalid_temperature)

print()


# ==================================================
# 5. VALIDATE MACHINE STRUCTURE
# ==================================================

def validate_machine(machine):
    """
    Check whether required machine fields exist.
    """

    required_fields = [
        "name",
        "temperature",
        "pressure",
        "rpm"
    ]

    for field in required_fields:

        if field not in machine:
            return False

    return True


print("TEST 4: Machine Validation")
print()


valid_machine = {
    "name": "Pump",
    "temperature": 85,
    "pressure": 120,
    "rpm": 1500
}


if validate_machine(valid_machine):
    print("Valid machine accepted.")

else:
    print("Valid machine rejected.")


invalid_machine = {
    "name": "Generator",
    "temperature": 105,
    "rpm": 3200
}


if validate_machine(invalid_machine):
    print("Invalid machine accepted.")

else:
    print("Invalid machine rejected.")

print()


# ==================================================
# 6. ANALYZE TEMPERATURE
# ==================================================

def analyze_temperature(temperature):

    if temperature >= 100:
        return "CRITICAL"

    elif temperature >= 80:
        return "HIGH"

    else:
        return "NORMAL"


# ==================================================
# 7. SAFE MACHINE ANALYSIS
# ==================================================

def analyze_machine(machine):
    """
    Safely analyze machine data.

    Returns a dictionary containing either
    a successful analysis or an error message.
    """

    # Check that all required fields exist
    if not validate_machine(machine):
        return {
            "status": "ERROR",
            "message": "Machine data is incomplete."
        }

    # Validate individual values
    temperature = validate_temperature(
        machine["temperature"]
    )

    pressure = validate_temperature(
        machine["pressure"]
    )

    rpm = validate_temperature(
        machine["rpm"]
    )

    # Check for invalid values
    if temperature is None:
        return {
            "status": "ERROR",
            "message": "Invalid temperature."
        }

    if pressure is None:
        return {
            "status": "ERROR",
            "message": "Invalid pressure."
        }

    if rpm is None:
        return {
            "status": "ERROR",
            "message": "Invalid RPM."
        }

    # Analyze temperature
    risk = analyze_temperature(temperature)

    return {
        "status": "SUCCESS",
        "machine": machine["name"],
        "temperature": temperature,
        "pressure": pressure,
        "rpm": rpm,
        "risk": risk
    }


# ==================================================
# 8. TEST VALID MACHINE
# ==================================================

print("TEST 5: Valid Machine Analysis")
print()


machine_1 = {
    "name": "Pump",
    "temperature": 85,
    "pressure": 120,
    "rpm": 1500
}


result = analyze_machine(machine_1)

print(result)

print()


# ==================================================
# 9. TEST INVALID MACHINE VALUES
# ==================================================

print("TEST 6: Invalid Machine Values")
print()


machine_2 = {
    "name": "Compressor",
    "temperature": "unknown",
    "pressure": 150,
    "rpm": 2800
}


result = analyze_machine(machine_2)

print(result)

print()


# ==================================================
# 10. TEST MISSING DATA
# ==================================================

print("TEST 7: Missing Machine Data")
print()


machine_3 = {
    "name": "Generator",
    "temperature": 105,
    "rpm": 3200
}


result = analyze_machine(machine_3)

print(result)

print()


# ==================================================
# 11. TRY / EXCEPT / ELSE / FINALLY
# ==================================================

print("TEST 8: Complete Error Handling")
print()


try:

    number = 100
    divisor = 5

    result = number / divisor

except ZeroDivisionError:

    print("Error: Cannot divide by zero.")

else:

    print("Calculation successful.")
    print("Result:", result)

finally:

    print("Calculation process finished.")


print()


# ==================================================
# 12. RAISING YOUR OWN ERROR
# ==================================================

def set_machine_temperature(temperature):
    """
    Set a machine temperature after validation.

    Raises:
        ValueError: If temperature is outside
        the allowed range.
    """

    temperature = float(temperature)

    if temperature < -50 or temperature > 200:
        raise ValueError(
            "Temperature is outside the allowed range."
        )

    return temperature


print("TEST 9: Custom Error")
print()


try:

    temperature = set_machine_temperature(150)

    print(
        "Accepted temperature:",
        temperature
    )

except (ValueError, TypeError) as error:

    print("Error:", error)


print()


# ==================================================
# 13. TEST OUT-OF-RANGE VALUE
# ==================================================

try:

    temperature = set_machine_temperature(250)

    print(
        "Accepted temperature:",
        temperature
    )

except (ValueError, TypeError) as error:

    print("Rejected temperature:", error)


print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 07 COMPLETE ===")
