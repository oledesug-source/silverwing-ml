# Silverwing ML
# Phase 1 - Lesson 09
# Object-Oriented Programming


print("=== SILVERWING ML ===")
print("Lesson 09: Object-Oriented Programming")
print()


# ==================================================
# 1. CREATE A CLASS
# ==================================================

class Machine:
    """
    Represents a machine.
    """

    def __init__(
            self,
            name,
            temperature,
            pressure,
            rpm
    ):
        self.name = name
        self.temperature = temperature
        self.pressure = pressure
        self.rpm = rpm


# ==================================================
# 2. CREATE A MACHINE OBJECT
# ==================================================

pump = Machine(
    "Pump",
    85,
    120,
    1500
)


print("TEST 1: Machine Object")
print()

print("Name:", pump.name)
print("Temperature:", pump.temperature)
print("Pressure:", pump.pressure)
print("RPM:", pump.rpm)

print()


# ==================================================
# 3. ADD METHODS
# ==================================================

class MachineAnalyzer:
    """
    Performs analysis on a machine.
    """

    def temperature_status(self, temperature):

        if temperature >= 100:
            return "CRITICAL"

        elif temperature >= 80:
            return "HIGH"

        else:
            return "NORMAL"


    def rpm_status(self, rpm):

        if rpm > 3000:
            return "HIGH"

        elif rpm > 2500:
            return "ELEVATED"

        else:
            return "NORMAL"


    def pressure_status(self, pressure):

        if pressure >= 160:
            return "CRITICAL"

        elif pressure >= 130:
            return "HIGH"

        else:
            return "NORMAL"


analyzer = MachineAnalyzer()


print("TEST 2: Machine Analyzer")
print()

print(
    "Temperature:",
    analyzer.temperature_status(
        pump.temperature
    )
)

print(
    "RPM:",
    analyzer.rpm_status(
        pump.rpm
    )
)

print(
    "Pressure:",
    analyzer.pressure_status(
        pump.pressure
    )
)

print()


# ==================================================
# 4. METHODS INSIDE THE MACHINE CLASS
# ==================================================

class SmartMachine:

    def __init__(
            self,
            name,
            temperature,
            pressure,
            rpm
    ):
        self.name = name
        self.temperature = temperature
        self.pressure = pressure
        self.rpm = rpm


    def temperature_status(self):

        if self.temperature >= 100:
            return "CRITICAL"

        elif self.temperature >= 80:
            return "HIGH"

        else:
            return "NORMAL"


    def rpm_status(self):

        if self.rpm > 3000:
            return "HIGH"

        elif self.rpm > 2500:
            return "ELEVATED"

        else:
            return "NORMAL"


    def pressure_status(self):

        if self.pressure >= 160:
            return "CRITICAL"

        elif self.pressure >= 130:
            return "HIGH"

        else:
            return "NORMAL"


    def risk_score(self):

        score = 0

        if self.temperature >= 100:
            score += 40

        elif self.temperature >= 80:
            score += 20


        if self.rpm > 3000:
            score += 40

        elif self.rpm > 2500:
            score += 15


        if self.pressure >= 160:
            score += 20

        elif self.pressure >= 130:
            score += 10


        return score


    def risk_level(self):

        score = self.risk_score()

        if score >= 70:
            return "CRITICAL"

        elif score >= 40:
            return "HIGH"

        elif score >= 20:
            return "MEDIUM"

        else:
            return "LOW"


    def display_status(self):

        print("Machine:", self.name)
        print(
            "Temperature:",
            self.temperature,
            "->",
            self.temperature_status()
        )

        print(
            "Pressure:",
            self.pressure,
            "->",
            self.pressure_status()
        )

        print(
            "RPM:",
            self.rpm,
            "->",
            self.rpm_status()
        )

        print(
            "Risk Score:",
            self.risk_score()
        )

        print(
            "Risk Level:",
            self.risk_level()
        )


# ==================================================
# 5. CREATE MULTIPLE OBJECTS
# ==================================================

pump = SmartMachine(
    "Pump",
    85,
    120,
    1500
)

compressor = SmartMachine(
    "Compressor",
    72,
    150,
    2800
)

generator = SmartMachine(
    "Generator",
    105,
    110,
    3200
)

turbine = SmartMachine(
    "Turbine",
    91,
    135,
    2900
)


machines = [
    pump,
    compressor,
    generator,
    turbine
]


# ==================================================
# 6. ANALYZE ALL MACHINES
# ==================================================

print("TEST 3: Smart Machines")
print()


for machine in machines:

    machine.display_status()

    print("-" * 40)


# ==================================================
# 7. FIND HIGHEST RISK
# ==================================================

highest_risk_machine = machines[0]


for machine in machines:

    if (
            machine.risk_score()
            >
            highest_risk_machine.risk_score()
    ):
        highest_risk_machine = machine


print()
print("TEST 4: Highest Risk Machine")
print()

print(
    "Machine:",
    highest_risk_machine.name
)

print(
    "Risk Score:",
    highest_risk_machine.risk_score()
)

print(
    "Risk Level:",
    highest_risk_machine.risk_level()
)

print()


# ==================================================
# 8. UNDERSTANDING OBJECTS
# ==================================================

print("TEST 5: Object Information")
print()

print(
    "Object type:",
    type(generator)
)

print(
    "Generator name:",
    generator.name
)

print(
    "Generator temperature:",
    generator.temperature
)

print()


# ==================================================
# 9. CHANGE OBJECT DATA
# ==================================================

print("TEST 6: Updating Object")
print()

print(
    "Original temperature:",
    pump.temperature
)

pump.temperature = 105

print(
    "New temperature:",
    pump.temperature
)

print(
    "New risk level:",
    pump.risk_level()
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 09 COMPLETE ===")
