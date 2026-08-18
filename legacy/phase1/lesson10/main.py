# Silverwing ML
# Phase 1 - Lesson 10
# Virtual Environments, Packages and Dependencies


import sys
import platform


print("=== SILVERWING ML ===")
print("Lesson 10: Environment and Dependencies")
print()


# ==================================================
# 1. PYTHON VERSION
# ==================================================

print("PYTHON INFORMATION")
print()

print("Python version:")
print(sys.version)

print()


# ==================================================
# 2. PYTHON EXECUTABLE
# ==================================================

print("PYTHON EXECUTABLE")
print()

print(sys.executable)

print()


# ==================================================
# 3. OPERATING SYSTEM
# ==================================================

print("SYSTEM INFORMATION")
print()

print("Operating system:")
print(platform.system())

print("System version:")
print(platform.version())

print("Machine architecture:")
print(platform.machine())

print()


# ==================================================
# 4. PYTHON IMPLEMENTATION
# ==================================================

print("PYTHON IMPLEMENTATION")
print()

print(platform.python_implementation())

print()


# ==================================================
# 5. CHECK A BUILT-IN PACKAGE
# ==================================================

print("STANDARD LIBRARY TEST")
print()

import json

sample_data = {
    "project": "Silverwing ML",
    "lesson": 10,
    "status": "running"
}

print(json.dumps(sample_data, indent=4))

print()


# ==================================================
# 6. CREATE A SIMPLE PACKAGE-STYLE FUNCTION
# ==================================================

def get_environment_info():

    return {
        "python_version": platform.python_version(),
        "python_implementation":
            platform.python_implementation(),
        "operating_system": platform.system(),
        "architecture": platform.machine()
    }


environment = get_environment_info()


print("ENVIRONMENT INFORMATION")
print()

for key, value in environment.items():

    print(
        key,
        ":",
        value
    )

print()


# ==================================================
# 7. CHECK VIRTUAL ENVIRONMENT
# ==================================================

print("VIRTUAL ENVIRONMENT")
print()

if hasattr(sys, "base_prefix"):

    if sys.prefix != sys.base_prefix:

        print("Virtual environment: ACTIVE")

    else:

        print("Virtual environment: NOT ACTIVE")

else:

    print("Unable to determine environment status.")

print()


# ==================================================
# 8. IMPORT AN OPTIONAL PACKAGE
# ==================================================

print("OPTIONAL PACKAGE TEST")
print()

try:

    import numpy

    print(
        "NumPy is installed."
    )

    print(
        "NumPy version:",
        numpy.__version__
    )

except ImportError:

    print(
        "NumPy is not installed yet."
    )

print()


# ==================================================
# 9. SIMPLE NUMPY TEST
# ==================================================

try:

    import numpy as np

    temperatures = np.array([
        72,
        85,
        91,
        68,
        105
    ])

    print("NUMPY DATA")

    print(
        "Temperatures:",
        temperatures
    )

    print(
        "Average:",
        temperatures.mean()
    )

    print(
        "Maximum:",
        temperatures.max()
    )

    print(
        "Minimum:",
        temperatures.min()
    )

except ImportError:

    print(
        "NumPy test skipped because "
        "NumPy is not installed."
    )

print()


# ==================================================
# 10. PROJECT INFORMATION
# ==================================================

project_info = {
    "name": "Silverwing ML",
    "phase": 1,
    "lesson": 10,
    "purpose":
        "Machine learning and communicative AI foundation"
}


print("PROJECT INFORMATION")
print()

for key, value in project_info.items():

    print(
        key,
        ":",
        value
    )

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 10 COMPLETE ===")
