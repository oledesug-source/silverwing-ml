# Silverwing ML
# Phase 1 - Lesson 12
# Testing and Quality Assurance


import unittest


print("=== SILVERWING ML ===")
print("Lesson 12: Testing")
print()


# ==================================================
# 1. FUNCTIONS WE WANT TO TEST
# ==================================================

def add_numbers(a, b):
    return a + b


def calculate_average(numbers):

    if not numbers:
        return 0

    return sum(numbers) / len(numbers)


def check_temperature(temperature):

    if temperature >= 100:
        return "CRITICAL"

    elif temperature >= 80:
        return "HIGH"

    else:
        return "NORMAL"


def calculate_risk(temperature, rpm):

    score = 0

    if temperature >= 100:
        score += 50

    elif temperature >= 80:
        score += 25

    if rpm > 3000:
        score += 50

    elif rpm > 2500:
        score += 20

    return score


# ==================================================
# 2. BASIC ASSERT TESTS
# ==================================================

print("TEST 1: Basic Assertions")
print()


assert add_numbers(2, 3) == 5

assert add_numbers(10, 20) == 30

assert calculate_average([10, 20, 30]) == 20

assert calculate_average([]) == 0

assert check_temperature(50) == "NORMAL"

assert check_temperature(85) == "HIGH"

assert check_temperature(110) == "CRITICAL"


print("All basic assertions passed.")
print()


# ==================================================
# 3. RISK TESTS
# ==================================================

print("TEST 2: Risk Calculations")
print()


assert calculate_risk(50, 1000) == 0

assert calculate_risk(85, 1500) == 25

assert calculate_risk(70, 2800) == 20

assert calculate_risk(105, 3200) == 100


print("All risk tests passed.")
print()


# ==================================================
# 4. UNITTEST TEST CLASS
# ==================================================

class TestSilverwingML(unittest.TestCase):

    def test_add_numbers(self):

        self.assertEqual(
            add_numbers(2, 3),
            5
        )


    def test_average(self):

        self.assertEqual(
            calculate_average([10, 20, 30]),
            20
        )


    def test_empty_average(self):

        self.assertEqual(
            calculate_average([]),
            0
        )


    def test_normal_temperature(self):

        self.assertEqual(
            check_temperature(70),
            "NORMAL"
        )


    def test_high_temperature(self):

        self.assertEqual(
            check_temperature(85),
            "HIGH"
        )


    def test_critical_temperature(self):

        self.assertEqual(
            check_temperature(105),
            "CRITICAL"
        )


    def test_low_risk(self):

        self.assertEqual(
            calculate_risk(70, 1500),
            0
        )


    def test_medium_risk(self):

        self.assertEqual(
            calculate_risk(85, 1500),
            25
        )


    def test_high_risk(self):

        self.assertEqual(
            calculate_risk(105, 1500),
            50
        )


    def test_critical_risk(self):

        self.assertEqual(
            calculate_risk(105, 3200),
            100
        )


# ==================================================
# 5. RUN UNITTESTS
# ==================================================

print("TEST 3: unittest Framework")
print()

if __name__ == "__main__":

    test_result = unittest.main(
        argv=["first-arg-is-ignored"],
        exit=False
    )

    print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 12 COMPLETE ===")
