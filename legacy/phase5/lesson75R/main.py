# Silverwing ML
# Phase 5 - Lesson 75R
# Silverwing Own Foundation Model
# Controlled Experiment Runner and Model Promotion
#
# Purpose:
# Build the infrastructure that turns evaluation into
# a controlled model-development loop:
#
# Current Model
#      ↓
# Experiment Definition
#      ↓
# Candidate Training
#      ↓
# Evaluation
#      ↓
# Comparison
#      ↓
# Promotion Gate
#      ↓
# Versioned Model Registry
#
# No external pretrained language model is used.

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LESSON_71_DIR = BASE_DIR.parent / "lesson71R"
LESSON_72_DIR = BASE_DIR.parent / "lesson72R"
LESSON_73_DIR = BASE_DIR.parent / "lesson73R"
LESSON_74_DIR = BASE_DIR.parent / "lesson74R"

MODEL_CONFIG_FILE = (
        LESSON_71_DIR / "silverwing_decoder_config.json"
)

DATASET_CONFIG_FILE = (
        LESSON_72_DIR / "silverwing_dataset_config.json"
)

TRAINING_LOG_FILE = (
        LESSON_73_DIR / "silverwing_training_log.json"
)

EVALUATION_REPORT = (
        LESSON_74_DIR / "silverwing_evaluation_report.json"
)

COMPARISON_REPORT = (
        LESSON_74_DIR / "silverwing_model_comparison.json"
)

BEST_CHECKPOINT = (
        LESSON_73_DIR
        / "checkpoints"
        / "silverwing_best.pt"
)

LATEST_CHECKPOINT = (
        LESSON_73_DIR
        / "checkpoints"
        / "silverwing_latest.pt"
)

EXPERIMENTS_DIR = BASE_DIR / "experiments"
REGISTRY_DIR = BASE_DIR / "registry"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_FILE = (
        REGISTRY_DIR / "silverwing_model_registry.json"
)

EXPERIMENT_LOG_FILE = (
        REGISTRY_DIR / "silverwing_experiment_log.json"
)

PROMOTION_LOG_FILE = (
        REGISTRY_DIR / "silverwing_promotion_log.json"
)

SEED = 42

MIN_REQUIRED_VALIDATION_IMPROVEMENT = 0.0


# ==================================================
# 2. HELPERS
# ==================================================

def utc_timestamp() -> float:
    return time.time()


def sha256_file(
        file_path: Path
) -> str:

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_json(
        file_path: Path
) -> Dict[str, Any]:

    with open(
            file_path,
            "r",
            encoding="utf-8"
    ) as file:
        return json.load(file)


def write_json(
        file_path: Path,
        data: Any
) -> None:

    with open(
            file_path,
            "w",
            encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def require_file(
        file_path: Path
) -> None:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{file_path}"
        )


# ==================================================
# 3. EXPERIMENT DATA STRUCTURES
# ==================================================

@dataclass
class ExperimentConfig:

    experiment_id: str

    model_family: str

    base_model_version: str

    dataset_version: str

    objective: str

    learning_rate: float

    epochs: int

    batch_size: int

    seed: int

    created_at: float

    status: str = "created"


@dataclass
class ModelVersion:

    version: str

    model_family: str

    checkpoint: str

    checkpoint_sha256: str

    validation_loss: Optional[float]

    perplexity: Optional[float]

    token_accuracy: Optional[float]

    dataset_version: str

    experiment_id: str

    created_at: float

    status: str


# ==================================================
# 4. MODEL REGISTRY
# ==================================================

class ModelRegistry:

    def __init__(
            self,
            registry_file: Path
    ):

        self.registry_file = registry_file

        self.data = {
            "active_version": None,
            "versions": [],
        }

        if registry_file.exists():
            self.data = read_json(
                registry_file
            )


    def save(self):

        write_json(
            self.registry_file,
            self.data
        )


    def active_version(self):

        return self.data.get(
            "active_version"
        )


    def versions(self):

        return self.data.get(
            "versions",
            []
        )


    def register(
            self,
            model_version: ModelVersion
    ):

        self.data.setdefault(
            "versions",
            []
        )

        self.data[
            "versions"
        ].append(
            asdict(
                model_version
            )
        )

        self.save()


    def promote(
            self,
            version: str
    ):

        found = False

        for item in self.data.get(
                "versions",
                []
        ):

            if item["version"] == version:

                item["status"] = "active"

                found = True

            elif item["status"] == "active":

                item["status"] = "previous"


        if not found:
            raise ValueError(
                f"Model version not found: {version}"
            )


        self.data[
            "active_version"
        ] = version

        self.save()


registry = ModelRegistry(
    REGISTRY_FILE
)


# ==================================================
# 5. VERIFY PREVIOUS LESSONS
# ==================================================

print("=== SILVERWING ML ===")
print("Phase 5 - Lesson 75R")
print("Controlled Experiment Runner and Model Promotion")
print()


print("TEST 1: Verify Previous Artifacts")
print()

required_artifacts = [
    MODEL_CONFIG_FILE,
    DATASET_CONFIG_FILE,
    TRAINING_LOG_FILE,
    EVALUATION_REPORT,
    COMPARISON_REPORT,
    BEST_CHECKPOINT,
    LATEST_CHECKPOINT,
]

for artifact in required_artifacts:
    require_file(artifact)

    print(
        "FOUND:",
        artifact
    )

print()


# ==================================================
# 6. LOAD PREVIOUS STATE
# ==================================================

print("TEST 2: Load Previous Training State")
print()

model_config = read_json(
    MODEL_CONFIG_FILE
)

dataset_config = read_json(
    DATASET_CONFIG_FILE
)

training_log = read_json(
    TRAINING_LOG_FILE
)

evaluation_report = read_json(
    EVALUATION_REPORT
)

comparison_report = read_json(
    COMPARISON_REPORT
)


print(
    "Model:",
    model_config.get("model")
)

print(
    "Dataset:",
    dataset_config.get("dataset")
)

print(
    "Latest training epochs:",
    training_log.get("epochs")
)

print(
    "Validation loss:",
    evaluation_report[
        "metrics"
    ].get(
        "validation_loss"
    )
)

print()


# ==================================================
# 7. CURRENT MODEL STATE
# ==================================================

current_validation_loss = (
    evaluation_report[
        "metrics"
    ].get(
        "validation_loss"
    )
)

current_perplexity = (
    evaluation_report[
        "metrics"
    ].get(
        "perplexity"
    )
)

current_accuracy = (
    evaluation_report[
        "metrics"
    ].get(
        "token_accuracy"
    )
)


print("TEST 3: Current Model Metrics")
print()

print(
    "Validation loss:",
    current_validation_loss
)

print(
    "Perplexity:",
    current_perplexity
)

print(
    "Token accuracy:",
    current_accuracy
)

print()


# ==================================================
# 8. INITIAL MODEL REGISTRATION
# ==================================================

print("TEST 4: Initialize Model Registry")
print()

if not registry.versions():

    initial_version = ModelVersion(
        version="silverwing-v0.1.0",

        model_family="Silverwing-Decoder",

        checkpoint=str(
            BEST_CHECKPOINT
        ),

        checkpoint_sha256=sha256_file(
            BEST_CHECKPOINT
        ),

        validation_loss=current_validation_loss,

        perplexity=current_perplexity,

        token_accuracy=current_accuracy,

        dataset_version=dataset_config.get(
            "dataset",
            "unknown"
        ),

        experiment_id="lesson74R-baseline",

        created_at=utc_timestamp(),

        status="active"
    )


    registry.register(
        initial_version
    )

    registry.promote(
        "silverwing-v0.1.0"
    )


    print(
        "Initial version registered."
    )

else:

    print(
        "Existing registry loaded."
    )


print(
    "Active version:",
    registry.active_version()
)

print()


# ==================================================
# 9. EXPERIMENT DEFINITION
# ==================================================

print("TEST 5: Create Experiment")
print()

experiment_id = (
    f"exp-{int(utc_timestamp())}"
)

experiment = ExperimentConfig(
    experiment_id=experiment_id,

    model_family="Silverwing-Decoder",

    base_model_version=(
            registry.active_version()
            or
            "silverwing-v0.1.0"
    ),

    dataset_version=dataset_config.get(
        "dataset",
        "Silverwing-Corpus-v1"
    ),

    objective=(
        "Improve validation loss without "
        "introducing numerical instability."
    ),

    learning_rate=3e-4,

    epochs=1,

    batch_size=2,

    seed=SEED,

    created_at=utc_timestamp()
)


experiment_path = (
        EXPERIMENTS_DIR
        / f"{experiment_id}.json"
)

write_json(
    experiment_path,
    asdict(experiment)
)

print(
    "Experiment ID:",
    experiment_id
)

print(
    "Base model:",
    experiment.base_model_version
)

print(
    "Dataset:",
    experiment.dataset_version
)

print(
    "Objective:",
    experiment.objective
)

print()


# ==================================================
# 10. EXPERIMENT ISOLATION
# ==================================================

print("TEST 6: Experiment Isolation")
print()

experiment_artifact_dir = (
        ARTIFACTS_DIR
        / experiment_id
)

experiment_artifact_dir.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "Experiment artifact directory:",
    experiment_artifact_dir
)

print(
    "Production model is not modified."
)

print()


# ==================================================
# 11. CANDIDATE METRIC SIMULATION
# ==================================================
#
# This lesson demonstrates the experiment and
# promotion infrastructure without silently retraining
# the production model.
#
# A real training run can later replace this section.


print("TEST 7: Candidate Evaluation Contract")
print()

candidate_validation_loss = current_validation_loss

candidate_perplexity = current_perplexity

candidate_accuracy = current_accuracy


candidate_metrics = {
    "validation_loss":
        candidate_validation_loss,

    "perplexity":
        candidate_perplexity,

    "token_accuracy":
        candidate_accuracy,

    "numerically_healthy":
        True
}


write_json(
    experiment_artifact_dir
    / "candidate_metrics.json",
    candidate_metrics
)


print(
    "Candidate metrics recorded."
)

print()


# ==================================================
# 12. COMPARISON
# ==================================================

print("TEST 8: Candidate vs Baseline")
print()

baseline_version = registry.active_version()

baseline_loss = current_validation_loss


loss_difference = (
        candidate_validation_loss
        -
        baseline_loss
)


improved = (
        candidate_validation_loss
        <
        (
                baseline_loss
                -
                MIN_REQUIRED_VALIDATION_IMPROVEMENT
        )
)


comparison = {
    "experiment_id":
        experiment_id,

    "baseline_version":
        baseline_version,

    "baseline_validation_loss":
        baseline_loss,

    "candidate_validation_loss":
        candidate_validation_loss,

    "loss_difference":
        loss_difference,

    "candidate_accuracy":
        candidate_accuracy,

    "candidate_perplexity":
        candidate_perplexity,

    "improved":
        improved
}


write_json(
    experiment_artifact_dir
    / "comparison.json",
    comparison
)


print(
    json.dumps(
        comparison,
        indent=4
    )
)

print()


# ==================================================
# 13. PROMOTION GATE
# ==================================================

print("TEST 9: Promotion Gate")
print()


def evaluate_promotion(
        metrics: Dict[str, Any],
        comparison_data: Dict[str, Any]
) -> Dict[str, Any]:

    if not metrics.get(
            "numerically_healthy",
            False
    ):
        return {
            "decision": "REJECT",
            "reason": (
                "Candidate failed numerical-health "
                "validation."
            ),
        }


    candidate_loss = (
        metrics.get(
            "validation_loss"
        )
    )

    baseline_loss = (
        comparison_data.get(
            "baseline_validation_loss"
        )
    )


    if (
            candidate_loss is None
            or
            baseline_loss is None
    ):
        return {
            "decision": "REJECT",
            "reason": (
                "Missing validation-loss metrics."
            ),
        }


    if (
            candidate_loss
            <
            baseline_loss
            -
            MIN_REQUIRED_VALIDATION_IMPROVEMENT
    ):
        return {
            "decision": "PROMOTE",
            "reason": (
                "Candidate improves validation loss."
            ),
        }


    return {
        "decision": "REJECT",
        "reason": (
            "Candidate did not improve validation loss."
        ),
    }


promotion_result = evaluate_promotion(
    candidate_metrics,
    comparison
)


print(
    "Decision:",
    promotion_result["decision"]
)

print(
    "Reason:",
    promotion_result["reason"]
)

print()


# ==================================================
# 14. PROMOTION ISOLATION
# ==================================================

print("TEST 10: Promotion Isolation")
print()

candidate_checkpoint = (
        experiment_artifact_dir
        / "candidate.pt"
)


if promotion_result[
    "decision"
] == "PROMOTE":

    shutil.copy2(
        BEST_CHECKPOINT,
        candidate_checkpoint
    )

    print(
        "Candidate checkpoint staged."
    )

else:

    print(
        "Candidate not promoted."
    )


print(
    "Active registry version remains:",
    registry.active_version()
)

print()


# ==================================================
# 15. VERSION GENERATION
# ==================================================

def next_patch_version(
        current_version: str
) -> str:

    if not current_version.startswith(
            "silverwing-v"
    ):

        raise ValueError(
            "Unexpected Silverwing version."
        )


    version_string = (
        current_version[
            len("silverwing-v"):
        ]
    )


    parts = version_string.split(
        "."
    )


    if len(parts) != 3:
        raise ValueError(
            "Version must use MAJOR.MINOR.PATCH."
        )


    major, minor, patch = (
        int(part)
        for part in parts
    )


    patch += 1


    return (
        f"silverwing-v"
        f"{major}.{minor}.{patch}"
    )


# ==================================================
# 16. REGISTER CANDIDATE
# ==================================================

print("TEST 11: Candidate Registration")
print()

candidate_version = next_patch_version(
    registry.active_version()
)


if (
        promotion_result["decision"]
        ==
        "PROMOTE"
):

    registered_candidate = ModelVersion(
        version=candidate_version,

        model_family="Silverwing-Decoder",

        checkpoint=str(
            candidate_checkpoint
        ),

        checkpoint_sha256=sha256_file(
            candidate_checkpoint
        ),

        validation_loss=candidate_validation_loss,

        perplexity=candidate_perplexity,

        token_accuracy=candidate_accuracy,

        dataset_version=experiment.dataset_version,

        experiment_id=experiment_id,

        created_at=utc_timestamp(),

        status="candidate"
    )


    registry.register(
        registered_candidate
    )


    print(
        "Candidate registered:",
        candidate_version
    )

else:

    print(
        "Candidate was not registered for promotion."
    )


print()


# ==================================================
# 17. MODEL PROMOTION
# ==================================================

print("TEST 12: Model Promotion")
print()


promotion_log_entry = {
    "experiment_id":
        experiment_id,

    "from_version":
        registry.active_version(),

    "candidate_version":
        (
            candidate_version
            if promotion_result[
                   "decision"
               ] == "PROMOTE"
            else None
        ),

    "decision":
        promotion_result["decision"],

    "reason":
        promotion_result["reason"],

    "timestamp":
        utc_timestamp()
}


if (
        promotion_result["decision"]
        ==
        "PROMOTE"
):

    registry.promote(
        candidate_version
    )

    promotion_log_entry[
        "active_version_after"
    ] = registry.active_version()

    print(
        "PROMOTED:",
        candidate_version
    )

else:

    promotion_log_entry[
        "active_version_after"
    ] = registry.active_version()

    print(
        "REJECTED:"
    )

    print(
        "Current model remains active."
    )


print()


# ==================================================
# 18. EXPERIMENT LOG
# ==================================================

print("TEST 13: Experiment Log")
print()

experiment_log = []

if EXPERIMENT_LOG_FILE.exists():

    experiment_log = read_json(
        EXPERIMENT_LOG_FILE
    )


if not isinstance(
        experiment_log,
        list
):

    experiment_log = []


experiment_log.append(
    {
        "experiment":
            asdict(experiment),

        "comparison":
            comparison,

        "promotion":
            promotion_result
    }
)


write_json(
    EXPERIMENT_LOG_FILE,
    experiment_log
)


print(
    "Experiment log entries:",
    len(experiment_log)
)

print()


# ==================================================
# 19. PROMOTION LOG
# ==================================================

print("TEST 14: Promotion Log")
print()

promotion_log = []

if PROMOTION_LOG_FILE.exists():

    promotion_log = read_json(
        PROMOTION_LOG_FILE
    )


if not isinstance(
        promotion_log,
        list
):

    promotion_log = []


promotion_log.append(
    promotion_log_entry
)


write_json(
    PROMOTION_LOG_FILE,
    promotion_log
)


print(
    "Promotion log entries:",
    len(promotion_log)
)

print()


# ==================================================
# 20. REGISTRY STATE
# ==================================================

print("TEST 15: Registry State")
print()

registry_state = {
    "active_version":
        registry.active_version(),

    "versions":
        registry.versions()
}


print(
    json.dumps(
        registry_state,
        indent=4
    )
)

print()


# ==================================================
# 21. VERSIONED ARTIFACT MANIFEST
# ==================================================

print("TEST 16: Artifact Manifest")
print()

manifest = {
    "model_config": {
        "path":
            str(MODEL_CONFIG_FILE),

        "sha256":
            sha256_file(
                MODEL_CONFIG_FILE
            )
    },

    "dataset_config": {
        "path":
            str(DATASET_CONFIG_FILE),

        "sha256":
            sha256_file(
                DATASET_CONFIG_FILE
            )
    },

    "training_log": {
        "path":
            str(TRAINING_LOG_FILE),

        "sha256":
            sha256_file(
                TRAINING_LOG_FILE
            )
    },

    "evaluation_report": {
        "path":
            str(EVALUATION_REPORT),

        "sha256":
            sha256_file(
                EVALUATION_REPORT
            )
    },

    "best_checkpoint": {
        "path":
            str(BEST_CHECKPOINT),

        "sha256":
            sha256_file(
                BEST_CHECKPOINT
            )
    }
}


write_json(
    experiment_artifact_dir
    / "artifact_manifest.json",
    manifest
)


print(
    "Manifest created."
)

print()


# ==================================================
# 22. RECOVERY / ROLLBACK
# ==================================================

print("TEST 17: Rollback Capability")
print()


def rollback_to_previous_active(
        registry_object: ModelRegistry
):

    versions = registry_object.versions()

    active = registry_object.active_version()


    if not active:
        raise RuntimeError(
            "No active version exists."
        )


    previous_versions = [
        item
        for item in versions
        if (
                item["version"] != active
                and
                item["status"] == "previous"
        )
    ]


    if not previous_versions:

        return None


    previous = previous_versions[-1]

    registry_object.promote(
        previous["version"]
    )

    return previous["version"]


print(
    "Rollback mechanism available."
)

print(
    "Current active version:",
    registry.active_version()
)

print()


# ==================================================
# 23. EXPERIMENT STATE MACHINE
# ==================================================

print("TEST 18: Experiment State Machine")
print()

states = [
    "CREATED",
    "PREPARING",
    "TRAINING",
    "EVALUATING",
    "COMPARING",
    "PROMOTING",
    "COMPLETED",
    "REJECTED",
]


for index, state in enumerate(states, start=1):

    print(
        f"{index}.",
        state
    )

print()


# ==================================================
# 24. CONTROLLED SELF-IMPROVEMENT LOOP
# ==================================================

print("CONTROLLED SELF-IMPROVEMENT LOOP")
print()

print("Current Silverwing")
print("        ↓")
print("Observe Performance")
print("        ↓")
print("Identify Weakness")
print("        ↓")
print("Create Experiment")
print("        ↓")
print("Candidate Training")
print("        ↓")
print("Evaluate")
print("        ↓")
print("Compare")
print("        ↓")
print("Promotion Gate")
print("     ┌──┴──┐")
print("     ↓     ↓")
print(" PROMOTE  REJECT")
print("     ↓     ↓")
print("Version   Preserve Current")
print("     ↓")
print("Next Cycle")

print()


# ==================================================
# 25. BIO-INSPIRED LOOP
# ==================================================

print("BIO-INSPIRED ADAPTATION LOOP")
print()

print("Environment / Experience")
print("          ↓")
print("Observation")
print("          ↓")
print("Internal Evaluation")
print("          ↓")
print("Adaptation Candidate")
print("          ↓")
print("Controlled Experiment")
print("          ↓")
print("Fitness / Performance")
print("          ↓")
print("Selection")
print("          ↓")
print("New Validated State")

print()


# ==================================================
# 26. IMPORTANT ENGINEERING PRINCIPLE
# ==================================================

print("IMPORTANT ENGINEERING PRINCIPLE")
print()

print(
    "Silverwing's autonomous growth must be "
    "versioned, measurable, reversible, and "
    "isolated from the active production model."
)

print()

print(
    "An experiment must never silently replace "
    "the active model."
)

print()

print(
    "Promotion is a separate explicit state "
    "transition after evaluation."
)

print()


# ==================================================
# 27. NEXT COMPONENT
# ==================================================

print("NEXT COMPONENT")
print()

print(
    "Silverwing now has:"
)

print(
    "training"
)

print(
    "evaluation"
)

print(
    "experiments"
)

print(
    "versioning"
)

print(
    "promotion gates"
)

print(
    "rollback infrastructure"
)

print()

print(
    "The next stage is to strengthen the actual "
    "learning curriculum so the native model learns "
    "general language patterns rather than only the "
    "tiny seed corpus."
)

print()


# ==================================================
# 28. FOUNDATION MODEL PROGRESS
# ==================================================

print("SILVERWING FOUNDATION MODEL PROGRESS")
print()

print("Own BPE Tokenizer")
print(" ↓")
print("Own Subword Vocabulary")
print(" ↓")
print("Own Token IDs")
print(" ↓")
print("Own Embedding System")
print(" ↓")
print("Own Position Encoding")
print(" ↓")
print("Own Self-Attention")
print(" ↓")
print("Own Transformer Block")
print(" ↓")
print("Own Decoder Language Model")
print(" ↓")
print("Own Training Data Pipeline")
print(" ↓")
print("Own Pretraining Engine")
print(" ↓")
print("Own Evaluation Framework")
print(" ↓")
print("OWN EXPERIMENT + PROMOTION SYSTEM")
print(" ↓")
print("Training Curriculum")
print(" ↓")
print("Instruction Training")
print(" ↓")
print("Reasoning Training")
print(" ↓")
print("Memory-Aware Training")
print(" ↓")
print("Continual Learning")
print(" ↓")
print("Controlled Autonomous Improvement")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 75R COMPLETE ===")