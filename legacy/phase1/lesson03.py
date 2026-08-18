# Silverwing ML
# Phase 1 - Lesson 03
# Python Data Structures


print("=== SILVERWING ML ===")
print("Lesson 03: Python Data Structures")
print()


# ==================================================
# 1. LISTS
# ==================================================

# A list stores multiple values.
machines = [
    "Pump",
    "Compressor",
    "Generator",
    "Conveyor"
]

print("MACHINES")
print(machines)
print()


# Access individual items
print("First machine:", machines[0])
print("Second machine:", machines[1])
print()


# Add a machine
machines.append("Turbine")

print("After adding a machine:")
print(machines)
print()


# Remove a machine
machines.remove("Conveyor")

print("After removing Conveyor:")
print(machines)
print()


# Number of machines
print("Number of machines:", len(machines))
print()


# Loop through the list
print("All machines:")

for machine in machines:
    print("-", machine)

print()


# ==================================================
# 2. MACHINE SENSOR DATA
# ==================================================

temperatures = [75, 82, 91, 68, 105]

print("TEMPERATURE DATA")
print(temperatures)
print()


print("Temperature readings:")

for temperature in temperatures:
    print(temperature)

print()


# Calculate total temperature
total_temperature = sum(temperatures)

print("Total temperature:", total_temperature)


# Calculate average temperature
average_temperature = total_temperature / len(temperatures)

print("Average temperature:", average_temperature)
print()


# ==================================================
# 3. DICTIONARIES
# ==================================================

# A dictionary stores information using key-value pairs.

machine = {
    "name": "Pump",
    "temperature": 85,
    "pressure": 120,
    "rpm": 1500,
    "operating_hours": 2500
}


print("MACHINE INFORMATION")
print(machine)
print()


# Access dictionary values
print("Machine name:", machine["name"])
print("Temperature:", machine["temperature"])
print("Pressure:", machine["pressure"])
print("RPM:", machine["rpm"])
print()


# Add new information
machine["location"] = "Factory A"

print("Machine location:", machine["location"])
print()


# ==================================================
# 4. LOOP THROUGH A DICTIONARY
# ==================================================

print("Machine data:")

for key, value in machine.items():
    print(key, ":", value)

print()


# ==================================================
# 5. TUPLES
# ==================================================

# A tuple is similar to a list,
# but its values cannot normally be changed.

coordinates = (10, 20)

print("Coordinates:", coordinates)
print("X:", coordinates[0])
print("Y:", coordinates[1])
print()


# ==================================================
# 6. SETS
# ==================================================

# A set stores unique values.

sensor_types = {
    "temperature",
    "pressure",
    "vibration",
    "temperature",
    "rpm"
}

print("Sensor types:")
print(sensor_types)
print()


# ==================================================
# 7. MACHINE DATA COLLECTION
# ==================================================

machines_data = [
    {
        "name": "Pump",
        "temperature": 85,
        "pressure": 120,
        "rpm": 1500
    },
    {
        "name": "Compressor",
        "temperature": 72,
        "pressure": 150,
        "rpm": 2800
    },
    {
        "name": "Generator",
        "temperature": 105,
        "pressure": 110,
        "rpm": 3200
    }
]


print("MACHINE DATA COLLECTION")
print()


for machine in machines_data:

    print("Machine:", machine["name"])
    print("Temperature:", machine["temperature"])
    print("Pressure:", machine["pressure"])
    print("RPM:", machine["rpm"])
    print()


# ==================================================
# 8. FIND HIGH-TEMPERATURE MACHINES
# ==================================================

print("HIGH TEMPERATURE MACHINES")
print()


for machine in machines_data:

    if machine["temperature"] >= 100:
        print(
            machine["name"],
            "has a critical temperature:",
            machine["temperature"]
        )


print()


# ==================================================
# 9. SIMPLE DATA ANALYSIS
# ==================================================

temperatures = []

for machine in machines_data:
    temperatures.append(machine["temperature"])


average_temperature = sum(temperatures) / len(temperatures)


print("All temperatures:", temperatures)
print("Average temperature:", average_temperature)
print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 03 COMPLETE ===")