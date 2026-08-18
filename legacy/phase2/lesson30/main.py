# Silverwing ML
# Phase 2 - Lesson 30
# Python Client -> ML API


import requests


print("=== SILVERWING ML ===")
print("Phase 2 - Lesson 30")
print("Python Client -> ML API")
print()


# ==================================================
# 1. API CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000"

PREDICT_URL = (
    f"{API_URL}/predict"
)

HEALTH_URL = (
    f"{API_URL}/health"
)

MODEL_URL = (
    f"{API_URL}/model"
)


# ==================================================
# 2. CHECK API CONNECTION
# ==================================================

print("TEST 1: Check API Connection")
print()


try:

    response = requests.get(
        HEALTH_URL,
        timeout=5
    )

    response.raise_for_status()

    health = response.json()

    print(
        "Health response:"
    )

    print(health)

except requests.RequestException as error:

    print(
        "Could not connect to ML API:"
    )

    print(error)

    print()

    print(
        "Make sure Lesson 29's "
        "Uvicorn server is running."
    )


print()


# ==================================================
# 3. GET MODEL INFORMATION
# ==================================================

print("TEST 2: Model Information")
print()


try:

    response = requests.get(
        MODEL_URL,
        timeout=5
    )

    response.raise_for_status()

    model_info = response.json()

    print(
        "Model information:"
    )

    print(model_info)

except requests.RequestException as error:

    print(
        "Could not retrieve model information:"
    )

    print(error)


print()


# ==================================================
# 4. PREPARE MACHINE DATA
# ==================================================

print("TEST 3: Prepare Machine Data")
print()


machine = {
    "temperature": 97,
    "pressure": 130,
    "rpm": 2600,
    "operating_hours": 3500
}


print(
    "Machine:"
)

print(machine)

print()


# ==================================================
# 5. SEND PREDICTION REQUEST
# ==================================================

print("TEST 4: Send Prediction Request")
print()


try:

    response = requests.post(
        PREDICT_URL,
        json=machine,
        timeout=10
    )

    response.raise_for_status()

    prediction_result = response.json()

    print(
        "API response:"
    )

    print(prediction_result)

except requests.RequestException as error:

    print(
        "Prediction request failed:"
    )

    print(error)

    prediction_result = None


print()


# ==================================================
# 6. READ PREDICTION
# ==================================================

if prediction_result is not None:

    print("TEST 5: Read Prediction")
    print()

    prediction = (
        prediction_result["prediction"]
    )

    confidence = (
        prediction_result["confidence"]
    )

    probabilities = (
        prediction_result["probabilities"]
    )


    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence
    )

    print()


    print(
        "Probabilities:"
    )


    for label, probability in (
            probabilities.items()
    ):

        print(
            label,
            ":",
            probability
        )


print()


# ==================================================
# 7. CREATE A HUMAN-READABLE RESULT
# ==================================================

if prediction_result is not None:

    print(
        "TEST 6: Human-Readable Interpretation"
    )

    print()


    if prediction == "CRITICAL":

        message = (
            "The machine is classified "
            "as CRITICAL. Immediate "
            "inspection is recommended."
        )

    elif prediction == "WARNING":

        message = (
            "The machine is classified "
            "as WARNING. Inspection "
            "should be considered."
        )

    else:

        message = (
            "The machine is classified "
            "as NORMAL."
        )


    print(message)

print()


# ==================================================
# 8. CREATE A SECOND MACHINE
# ==================================================

print("TEST 7: Multiple API Requests")
print()


machines = [

    {
        "name": "Pump",
        "temperature": 75,
        "pressure": 110,
        "rpm": 1500,
        "operating_hours": 1500
    },

    {
        "name": "Compressor",
        "temperature": 88,
        "pressure": 125,
        "rpm": 2400,
        "operating_hours": 2800
    },

    {
        "name": "Generator",
        "temperature": 105,
        "pressure": 140,
        "rpm": 3200,
        "operating_hours": 4500
    }
]


results = []


for machine_data in machines:

    machine_name = machine_data["name"]


    payload = {
        "temperature":
            machine_data["temperature"],

        "pressure":
            machine_data["pressure"],

        "rpm":
            machine_data["rpm"],

        "operating_hours":
            machine_data["operating_hours"]
    }


    try:

        response = requests.post(
            PREDICT_URL,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()


        results.append({
            "name": machine_name,
            "prediction":
                result["prediction"],
            "confidence":
                result["confidence"]
        })


    except requests.RequestException as error:

        print(
            "Failed to analyze",
            machine_name
        )

        print(error)


print(
    "API analysis results:"
)

print()


for result in results:

    print(
        result["name"],
        "->",
        result["prediction"],
        "| confidence:",
        round(
            result["confidence"],
            4
        )
    )


print()


# ==================================================
# 9. BASIC API CLIENT CLASS
# ==================================================

print("TEST 8: Build Reusable API Client")
print()


class SilverwingMLClient:
    """
    Client for communicating with
    the Silverwing ML API.
    """

    def __init__(self, base_url):

        self.base_url = base_url.rstrip("/")


    def health(self):

        response = requests.get(
            f"{self.base_url}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()


    def model_info(self):

        response = requests.get(
            f"{self.base_url}/model",
            timeout=5
        )

        response.raise_for_status()

        return response.json()


    def predict(
            self,
            temperature,
            pressure,
            rpm,
            operating_hours
    ):

        payload = {
            "temperature": temperature,
            "pressure": pressure,
            "rpm": rpm,
            "operating_hours":
                operating_hours
        }


        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        return response.json()


# ==================================================
# 10. USE CLIENT CLASS
# ==================================================

client = SilverwingMLClient(
    API_URL
)


print(
    "Client health:"
)

print(
    client.health()
)

print()


print(
    "Client model information:"
)

print(
    client.model_info()
)

print()


client_result = client.predict(
    temperature=92,
    pressure=128,
    rpm=2500,
    operating_hours=3200
)


print(
    "Client prediction:"
)

print(
    client_result
)

print()


# ==================================================
# 11. API COMMUNICATION PIPELINE
# ==================================================

print("API COMMUNICATION PIPELINE")
print()

print("Python Client")
print("     ↓")
print("HTTP POST /predict")
print("     ↓")
print("FastAPI")
print("     ↓")
print("ML Model")
print("     ↓")
print("Prediction")
print("     ↓")
print("JSON Response")
print("     ↓")
print("Python Client")

print()


# ==================================================
# 12. FUTURE AI ARCHITECTURE
# ==================================================

print("FUTURE SILVERWING ARCHITECTURE")
print()

print("User")
print(" ↓")
print("Communicative AI")
print(" ↓")
print("Reasoning / Agent")
print(" ↓")
print("ML Client")
print(" ↓")
print("Silverwing ML API")
print(" ↓")
print("Prediction")
print(" ↓")
print("AI Interpretation")
print(" ↓")
print("Natural-Language Response")

print()


# ==================================================
# 13. ERROR HANDLING PRINCIPLE
# ==================================================

print("API ERROR-HANDLING PRINCIPLE")
print()

print(
    "A client should never assume "
    "that an API is always available."
)

print()

print(
    "Production systems need:"
)

print(
    "- timeouts"
)

print(
    "- HTTP error handling"
)

print(
    "- retries where appropriate"
)

print(
    "- logging"
)

print(
    "- service health checks"
)

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 30 COMPLETE ===")
