# Silverwing ML
# Phase 2 - Lesson 29
# ML Prediction API with FastAPI


from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
        BASE_DIR
        / "silverwing_risk_model.joblib"
)


# ==================================================
# 2. TRAINING DATA
# ==================================================

data = {
    "temperature": [
        60, 62, 65, 68, 70,
        72, 74, 76, 78, 80,
        82, 84, 86, 88, 90,
        92, 94, 96, 98, 100,
        102, 104, 106, 108, 110
    ],

    "pressure": [
        95, 98, 100, 102, 105,
        108, 110, 112, 115, 118,
        120, 122, 124, 126, 128,
        130, 132, 134, 136, 138,
        140, 142, 144, 146, 148
    ],

    "rpm": [
        1200, 1250, 1300, 1350, 1400,
        1450, 1500, 1550, 1600, 1700,
        1800, 1900, 2000, 2100, 2200,
        2300, 2400, 2500, 2600, 2750,
        2900, 3000, 3100, 3200, 3400
    ],

    "operating_hours": [
        200, 400, 600, 800, 1000,
        1200, 1400, 1600, 1800, 2000,
        2200, 2400, 2600, 2800, 3000,
        3200, 3400, 3600, 3800, 4000,
        4200, 4400, 4600, 4800, 5000
    ],

    "risk_level": [
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",
        "NORMAL",

        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",
        "WARNING",

        "CRITICAL",
        "CRITICAL",
        "CRITICAL",
        "CRITICAL",
        "CRITICAL"
    ]
}


df = pd.DataFrame(data)


FEATURE_COLUMNS = [
    "temperature",
    "pressure",
    "rpm",
    "operating_hours"
]


# ==================================================
# 3. TRAIN OR LOAD MODEL
# ==================================================

def train_model():
    """
    Train the Random Forest model and save it.
    """

    X = df[FEATURE_COLUMNS]

    y = df["risk_level"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    return model


def load_model():
    """
    Load an existing model or train one if
    the model artifact does not exist.
    """

    if MODEL_FILE.exists():

        return joblib.load(
            MODEL_FILE
        )

    return train_model()


model = load_model()


# ==================================================
# 4. CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Silverwing ML API",
    description=(
        "Machine-learning prediction service "
        "for Silverwing."
    ),
    version="1.0.0"
)


# ==================================================
# 5. REQUEST MODEL
# ==================================================

class MachineInput(BaseModel):

    temperature: float
    pressure: float
    rpm: float
    operating_hours: float


# ==================================================
# 6. ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "name": "Silverwing ML API",
        "version": "1.0.0",
        "status": "online",
        "model": type(model).__name__
    }


# ==================================================
# 7. HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health():

    model_loaded = (
            model is not None
    )

    return {
        "status": "healthy"
        if model_loaded
        else "unhealthy",
        "model_loaded": model_loaded
    }


# ==================================================
# 8. MODEL INFORMATION
# ==================================================

@app.get("/model")
def model_information():

    return {
        "model_type": type(model).__name__,
        "features": FEATURE_COLUMNS,
        "classes": (
            model.classes_.tolist()
        ),
        "model_file": MODEL_FILE.name
    }


# ==================================================
# 9. PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
def predict(machine: MachineInput):

    try:

        input_data = pd.DataFrame([
            {
                "temperature":
                    machine.temperature,

                "pressure":
                    machine.pressure,

                "rpm":
                    machine.rpm,

                "operating_hours":
                    machine.operating_hours
            }
        ])

        prediction = model.predict(
            input_data
        )[0]

        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )

        probability_map = {
            label: float(probability)
            for label, probability in zip(
                model.classes_,
                probabilities
            )
        }

        confidence = max(
            probability_map.values()
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probability_map,
            "input": {
                "temperature":
                    machine.temperature,

                "pressure":
                    machine.pressure,

                "rpm":
                    machine.rpm,

                "operating_hours":
                    machine.operating_hours
            }
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==================================================
# 10. LOCAL TEST FUNCTION
# ==================================================

def local_prediction_test():

    sample = pd.DataFrame([
        {
            "temperature": 97,
            "pressure": 130,
            "rpm": 2600,
            "operating_hours": 3500
        }
    ])

    prediction = model.predict(
        sample
    )[0]

    probabilities = model.predict_proba(
        sample
    )[0]

    print()
    print("=== LOCAL MODEL TEST ===")
    print()

    print(
        "Input:",
        sample.to_dict(
            orient="records"
        )[0]
    )

    print()

    print(
        "Prediction:",
        prediction
    )

    print()

    print("Probabilities:")

    for label, probability in zip(
            model.classes_,
            probabilities
    ):

        print(
            label,
            ":",
            round(
                float(probability),
                4
            )
        )

    print()


# ==================================================
# 11. DIRECT EXECUTION
# ==================================================

if __name__ == "__main__":

    local_prediction_test()

    print(
        "FastAPI application created."
    )

    print(
        "Start the API with:"
    )

    print()

    print(
        "python -m uvicorn "
        "main:app --reload"
    )
