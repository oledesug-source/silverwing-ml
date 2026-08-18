# Silverwing ML
# Phase 1 - Lesson 11
# Dependency Management


import sys
import subprocess


print("=== SILVERWING ML ===")
print("Lesson 11: Dependency Management")
print()


# ==================================================
# 1. DISPLAY PYTHON ENVIRONMENT
# ==================================================

print("PYTHON ENVIRONMENT")
print()

print("Python executable:")
print(sys.executable)

print()


# ==================================================
# 2. IMPORT NUMPY
# ==================================================

try:

    import numpy as np

    print("NumPy successfully imported.")
    print("NumPy version:", np.__version__)

except ImportError:

    print("NumPy is not installed.")

print()


# ==================================================
# 3. CREATE NUMERICAL DATA
# ==================================================

temperatures = np.array([
    72,
    85,
    91,
    68,
    105
])


print("TEMPERATURE DATA")
print()

print("Data:", temperatures)
print("Average:", temperatures.mean())
print("Maximum:", temperatures.max())
print("Minimum:", temperatures.min())

print()


# ==================================================
# 4. CHECK INSTALLED PACKAGE
# ==================================================

print("PACKAGE INFORMATION")
print()


try:

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "show",
            "numpy"
        ],
        capture_output=True,
        text=True,
        check=False
    )

    print(result.stdout)

except Exception as error:

    print(
        "Unable to retrieve package information:",
        error
    )


# ==================================================
# 5. PROJECT DEPENDENCY MESSAGE
# ==================================================

print("PROJECT DEPENDENCIES")
print()

print(
    "This project currently requires:"
)

print(
    "NumPy 2.5.1"
)

print()


# ==================================================
# 6. WHY REQUIREMENTS.TXT MATTERS
# ==================================================

print("DEPENDENCY PURPOSE")
print()

print(
    "requirements.txt allows another "
    "environment to install the project's "
    "required Python packages."
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 11 COMPLETE ===")
