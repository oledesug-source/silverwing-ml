# Silverwing ML
# Phase 1 - Lesson 08
# Files and JSON


import json


print("=== SILVERWING ML ===")
print("Lesson 08: Files and JSON")
print()


# ==================================================
# 1. WRITE TEXT TO A FILE
# ==================================================

print("TEST 1: Writing a File")
print()


message = "Silverwing ML is learning how to store data."

with open("silverwing.txt", "w", encoding="utf-8") as file:
    file.write(message)


print("Text file created successfully.")
print()


# ==================================================
# 2. READ TEXT FROM A FILE
# ==================================================

print("TEST 2: Reading a File")
print()


with open("silverwing.txt", "r", encoding="utf-8") as file:
    content = file.read()


print("File content:")
print(content)
print()


# ==================================================
# 3. MACHINE DATA
# ==================================================

machine = {
    "name": "Pump",
    "temperature": 85,
    "pressure": 120,
    "rpm": 1500,
    "operating_hours": 2500
}


print("TEST 3: Machine Data")
print()

print(machine)
print()


# ==================================================
# 4. SAVE MACHINE DATA AS JSON
# ==================================================

print("TEST 4: Saving JSON")
print()


with open("machine.json", "w", encoding="utf-8") as file:

    json.dump(
        machine,
        file,
        indent=4
    )


print("Machine data saved to machine.json.")
print()


# ==================================================
# 5. READ JSON
# ==================================================

print("TEST 5: Reading JSON")
print()


with open("machine.json", "r", encoding="utf-8") as file:

    loaded_machine = json.load(file)


print("Loaded machine:")
print(loaded_machine)
print()


# ==================================================
# 6. ACCESS JSON DATA
# ==================================================

print("TEST 6: Accessing JSON Data")
print()


print("Machine name:", loaded_machine["name"])
print("Temperature:", loaded_machine["temperature"])
print("Pressure:", loaded_machine["pressure"])
print("RPM:", loaded_machine["rpm"])

print()


# ==================================================
# 7. SAVE MULTIPLE MACHINES
# ==================================================

machines = [
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
    },

    {
        "name": "Turbine",
        "temperature": 91,
        "pressure": 135,
        "rpm": 2900
    }
]


print("TEST 7: Saving Multiple Machines")
print()


with open("machines.json", "w", encoding="utf-8") as file:

    json.dump(
        machines,
        file,
        indent=4
    )


print("Machine collection saved.")
print()


# ==================================================
# 8. LOAD MULTIPLE MACHINES
# ==================================================

print("TEST 8: Loading Multiple Machines")
print()


with open("machines.json", "r", encoding="utf-8") as file:

    loaded_machines = json.load(file)


for machine in loaded_machines:

    print(
        machine["name"],
        "->",
        machine["temperature"],
        "°C"
    )


print()


# ==================================================
# 9. UPDATE DATA
# ==================================================

print("TEST 9: Updating Data")
print()


loaded_machine["temperature"] = 92

loaded_machine["status"] = "HIGH"


with open("machine.json", "w", encoding="utf-8") as file:

    json.dump(
        loaded_machine,
        file,
        indent=4
    )


print("Machine data updated.")
print()


# ==================================================
# 10. LOAD UPDATED DATA
# ==================================================

print("TEST 10: Confirming Update")
print()


with open("machine.json", "r", encoding="utf-8") as file:

    updated_machine = json.load(file)


print("Updated machine:")
print(updated_machine)
print()


# ==================================================
# 11. SIMPLE MEMORY SYSTEM
# ==================================================

memory = {
    "user_name": "User",
    "last_machine_checked": "Pump",
    "last_temperature": 92,
    "last_status": "HIGH"
}


print("TEST 11: Simple AI Memory")
print()


with open("memory.json", "w", encoding="utf-8") as file:

    json.dump(
        memory,
        file,
        indent=4
    )


print("Memory saved.")
print()


# ==================================================
# 12. LOAD MEMORY
# ==================================================

with open("memory.json", "r", encoding="utf-8") as file:

    loaded_memory = json.load(file)


print("AI memory:")
print()


for key, value in loaded_memory.items():

    print(
        key,
        ":",
        value
    )


print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 08 COMPLETE ===")
