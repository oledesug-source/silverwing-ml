# Silverwing ML
# Phase 3 - Lesson 33
# Multi-Layer Neural Networks and Activation Functions


import torch
import torch.nn as nn
import torch.optim as optim


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 33")
print("Multi-Layer Neural Networks")
print()


# ==================================================
# 1. PYTORCH INFORMATION
# ==================================================

print("TEST 1: PyTorch Information")
print()

print("PyTorch version:", torch.__version__)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print()


# ==================================================
# 2. CREATE TRAINING DATA
# ==================================================

print("TEST 2: Training Dataset")
print()


# Four machine features:
#
# temperature
# pressure
# rpm
# operating_hours

X = torch.tensor([
    [60.0, 95.0, 1200.0, 200.0],
    [65.0, 100.0, 1300.0, 600.0],
    [70.0, 105.0, 1400.0, 1000.0],
    [75.0, 110.0, 1600.0, 1400.0],
    [80.0, 115.0, 1800.0, 1800.0],
    [85.0, 120.0, 2000.0, 2200.0],
    [90.0, 125.0, 2200.0, 2600.0],
    [95.0, 130.0, 2500.0, 3200.0],
    [100.0, 135.0, 2800.0, 3800.0],
    [105.0, 140.0, 3000.0, 4300.0],
    [110.0, 145.0, 3200.0, 4800.0]
])


# Risk score target

y = torch.tensor([
    [0.0],
    [5.0],
    [10.0],
    [15.0],
    [20.0],
    [25.0],
    [30.0],
    [45.0],
    [60.0],
    [80.0],
    [100.0]
])


print("Input shape:", X.shape)
print("Target shape:", y.shape)

print()


# ==================================================
# 3. NORMALIZE INPUT FEATURES
# ==================================================

print("TEST 3: Feature Normalization")
print()


X_mean = X.mean(
    dim=0,
    keepdim=True
)

X_std = X.std(
    dim=0,
    keepdim=True
)


X_normalized = (
                       X - X_mean
               ) / (
                       X_std + 1e-8
               )


print(
    "Normalized features:"
)

print(
    X_normalized
)

print()


# ==================================================
# 4. CREATE MULTI-LAYER NETWORK
# ==================================================

print("TEST 4: Multi-Layer Neural Network")
print()


class DeepMachineRiskNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input layer
            nn.Linear(
                4,
                32
            ),

            # First activation
            nn.ReLU(),

            # Hidden layer
            nn.Linear(
                32,
                16
            ),

            # Second activation
            nn.ReLU(),

            # Another hidden layer
            nn.Linear(
                16,
                8
            ),

            # Third activation
            nn.ReLU(),

            # Output layer
            nn.Linear(
                8,
                1
            )
        )


    def forward(self, x):

        return self.network(x)


model = DeepMachineRiskNetwork()


print(model)

print()


# ==================================================
# 5. COUNT PARAMETERS
# ==================================================

print("TEST 5: Trainable Parameters")
print()


total_parameters = 0


for name, parameter in (
        model.named_parameters()
):

    parameter_count = parameter.numel()

    total_parameters += parameter_count

    print(
        name,
        "->",
        parameter.shape,
        "| parameters:",
        parameter_count
    )


print()

print(
    "Total trainable parameters:",
    total_parameters
)

print()


# ==================================================
# 6. TEST DIFFERENT ACTIVATION FUNCTIONS
# ==================================================

print("TEST 6: Activation Functions")
print()


values = torch.tensor([
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0
])


relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


print("Input:")
print(values)

print()

print("ReLU:")
print(relu(values))

print()

print("Sigmoid:")
print(sigmoid(values))

print()

print("Tanh:")
print(tanh(values))

print()


# ==================================================
# 7. WHY ACTIVATION FUNCTIONS MATTER
# ==================================================

print("TEST 7: Activation Function Purpose")
print()

print(
    "Activation functions introduce non-linearity."
)

print()

print(
    "Without non-linear activations, stacking "
    "linear layers would still behave like "
    "a single linear transformation."
)

print()


# ==================================================
# 8. LOSS FUNCTION
# ==================================================

print("TEST 8: Loss Function")
print()


loss_function = nn.MSELoss()


print(
    "Loss function:",
    type(loss_function).__name__
)

print()


# ==================================================
# 9. OPTIMIZER
# ==================================================

print("TEST 9: Optimizer")
print()


optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


print(
    "Optimizer:",
    type(optimizer).__name__
)

print()


# ==================================================
# 10. INITIAL PREDICTIONS
# ==================================================

print("TEST 10: Initial Predictions")
print()


with torch.no_grad():

    initial_predictions = model(
        X_normalized
    )


print(
    initial_predictions
)

print()


# ==================================================
# 11. TRAIN NETWORK
# ==================================================

print("TEST 11: Neural Network Training")
print()


epochs = 1500


for epoch in range(epochs):

    # ----------------------------------------------
    # Forward pass
    # ----------------------------------------------

    predictions = model(
        X_normalized
    )


    # ----------------------------------------------
    # Calculate loss
    # ----------------------------------------------

    loss = loss_function(
        predictions,
        y
    )


    # ----------------------------------------------
    # Clear previous gradients
    # ----------------------------------------------

    optimizer.zero_grad()


    # ----------------------------------------------
    # Backpropagation
    # ----------------------------------------------

    loss.backward()


    # ----------------------------------------------
    # Update weights
    # ----------------------------------------------

    optimizer.step()


    if (
            epoch == 0
            or
            (epoch + 1) % 150 == 0
    ):

        print(
            "Epoch:",
            epoch + 1,
            "| Loss:",
            loss.item()
        )


print()


# ==================================================
# 12. FINAL PREDICTIONS
# ==================================================

print("TEST 12: Final Predictions")
print()


with torch.no_grad():

    final_predictions = model(
        X_normalized
    )


for actual, predicted in zip(
        y,
        final_predictions
):

    print(
        "Actual:",
        round(
            actual.item(),
            2
        ),
        "| Predicted:",
        round(
            predicted.item(),
            2
        )
    )


print()


# ==================================================
# 13. FINAL LOSS
# ==================================================

print("TEST 13: Final Loss")
print()


with torch.no_grad():

    final_loss = loss_function(
        final_predictions,
        y
    )


print(
    "Final MSE:",
    final_loss.item()
)

print()


# ==================================================
# 14. NEW MACHINE PREDICTION
# ==================================================

print("TEST 14: New Machine")
print()


new_machine = torch.tensor([
    [
        97.0,
        130.0,
        2600.0,
        3500.0
    ]
])


new_machine_normalized = (
                                 new_machine - X_mean
                         ) / (
                                 X_std + 1e-8
                         )


with torch.no_grad():

    new_prediction = model(
        new_machine_normalized
    )


print(
    "Machine:"
)

print(
    new_machine
)

print()

print(
    "Predicted risk score:",
    new_prediction.item()
)

print()


# ==================================================
# 15. INSPECT HIDDEN LAYERS
# ==================================================

print("TEST 15: Hidden Layer Sizes")
print()


print(
    "Input:",
    4
)

print(
    "Hidden Layer 1:",
    32
)

print(
    "Hidden Layer 2:",
    16
)

print(
    "Hidden Layer 3:",
    8
)

print(
    "Output:",
    1
)

print()


# ==================================================
# 16. NETWORK ARCHITECTURE
# ==================================================

print("NETWORK ARCHITECTURE")
print()

print("4 input features")
print("       ↓")
print("Linear(4 → 32)")
print("       ↓")
print("ReLU")
print("       ↓")
print("Linear(32 → 16)")
print("       ↓")
print("ReLU")
print("       ↓")
print("Linear(16 → 8)")
print("       ↓")
print("ReLU")
print("       ↓")
print("Linear(8 → 1)")
print("       ↓")
print("Risk prediction")

print()


# ==================================================
# 17. DEEP LEARNING CONCEPT
# ==================================================

print("DEEP LEARNING CONCEPT")
print()

print(
    "A network becomes deeper when it contains "
    "multiple learnable layers."
)

print()

print(
    "Each layer can transform the representation "
    "of the information before passing it onward."
)

print()


# ==================================================
# 18. REPRESENTATION LEARNING
# ==================================================

print("REPRESENTATION LEARNING")
print()

print(
    "Earlier layers can learn useful intermediate "
    "representations of the input."
)

print()

print(
    "Later layers can combine those representations "
    "to produce a final prediction."
)

print()


# ==================================================
# 19. CONNECTION TO LANGUAGE MODELS
# ==================================================

print("CONNECTION TO LANGUAGE MODELS")
print()

print(
    "Modern language models also contain many "
    "layers that transform representations."
)

print()

print(
    "However, modern LLMs use Transformer "
    "architectures with attention mechanisms."
)

print()

print(
    "Our current network is intentionally much "
    "simpler so the learning mechanics remain clear."
)

print()


# ==================================================
# 20. CURRENT SILVERWING PIPELINE
# ==================================================

print("CURRENT SILVERWING DEEP LEARNING PIPELINE")
print()

print("Machine Data")
print("      ↓")
print("Normalization")
print("      ↓")
print("Multi-Layer Network")
print("      ↓")
print("Activation Functions")
print("      ↓")
print("Prediction")
print("      ↓")
print("Loss")
print("      ↓")
print("Backpropagation")
print("      ↓")
print("Parameter Updates")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 33 COMPLETE ===")
