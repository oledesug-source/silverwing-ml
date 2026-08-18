# Silverwing ML
# Phase 1 - Lesson 06
# Machine Module


machines = [
    {
        "name": "Pump",
        "temperature": 85,
        "pressure": 120,
        "rpm": 1500,
        "operating_hours": 2500
    },

    {
        "name": "Compressor",
        "temperature": 72,
        "pressure": 150,
        "rpm": 2800,
        "operating_hours": 3200
    },

    {
        "name": "Generator",
        "temperature": 105,
        "pressure": 110,
        "rpm": 3200,
        "operating_hours": 4500
    },

    {
        "name": "Turbine",
        "temperature": 91,
        "pressure": 135,
        "rpm": 2900,
        "operating_hours": 3800
    }
]


def get_machines():
    return machines


def get_machine_count():
    return len(machines)