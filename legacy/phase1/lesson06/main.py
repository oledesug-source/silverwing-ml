# Silverwing ML
# Phase 1 - Lesson 06
# Main Program


from machine import get_machines
from analysis import analyze_machine


print("=== SILVERWING ML ===")
print("Lesson 06: Modules and Imports")
print()


machines = get_machines()


print("Number of machines:", len(machines))
print()


results = []


for machine in machines:

    result = analyze_machine(machine)

    results.append(result)


print("MACHINE ANALYSIS")
print()


for result in results:

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


print()
print("=== LESSON 06 COMPLETE ===")