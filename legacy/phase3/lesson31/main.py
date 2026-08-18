# Silverwing ML
# Phase 3 - Lesson 31
# Introduction to Neural Networks with PyTorch


import torch
import torch.nn as nn
import torch.optim as optim


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 31")
print("Neural Networks with PyTorch")
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
# 2. CREATE TENSORS
# ==================================================

print("TEST 2: Tensors")
print()

temperatures = torch.tensor([
    70.0,
    75.0,
    80.0,
    85.0,
    90.0
])


print("Temperature tensor:")
print(temperatures)

print()

print("Tensor shape:")
print(temperatures.shape)

print()


# ==================================================
# 3. TENSOR OPERATIONS
# ==================================================

print("TEST 3: Tensor Operations")
print()

print(
    "Temperatures + 5:"
)

print(
    temperatures + 5
)

print()

print(
    "Temperatures × 2:"
)

print(
    temperatures * 2
)

print()


# ==================================================
# 4. MACHINE FEATURES
# ==================================================

print("TEST 4: Machine Feature Tensor")
print()


machine_features = torch.tensor([
    [
        85.0,
        120.0,
        1500.0,
        2500.0
    ],

    [
        72.0,
        150.0,
        2800.0,
        3200.0
    ],

    [
        105.0,
        110.0,
        3200.0,
        4500.0
    ],

    [
        91.0,
        135.0,
        2900.0,
        3800.0
    ]
])


print(machine_features)

print()

print(
    "Shape:",
    machine_features.shape
)

print()


# ==================================================
# 5. CREATE A SIMPLE NEURAL NETWORK
# ==================================================

print("TEST 5: Neural Network")
print()


class MachineRiskNetwork(nn.Module):
    """
    Simple feed-forward neural network.

    Input:
        4 machine features

    Output:
        1 predicted risk value
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                4,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                8
            ),

            nn.ReLU(),

            nn.Linear(
                8,
                1
            )
        )


    def forward(self, x):

        return self.network(x)


model = MachineRiskNetwork()


print(model)

print()


# ==================================================
# 6. CREATE TRAINING DATA
# ==================================================

print("TEST 6: Training Data")
print()


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


print(
    "Input shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)

print()


# ==================================================
# 7. IMPORTANT: FEATURE NORMALIZATION
# ==================================================

print("TEST 7: Normalize Features")
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
    "Normalized data:"
)

print(
    X_normalized
)

print()


# ==================================================
# 8. CREATE LOSS FUNCTION
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
# 9. CREATE OPTIMIZER
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
# 10. TRAINING LOOP
# ==================================================

print("TEST 10: Training Neural Network")
print()


epochs = 1000


for epoch in range(epochs):

    # Forward pass

    predictions = model(
        X_normalized
    )


    # Calculate loss

    loss = loss_function(
        predictions,
        y
    )


    # Clear old gradients

    optimizer.zero_grad()


    # Backpropagation

    loss.backward()


    # Update model parameters

    optimizer.step()


    # Display progress

    if (
            epoch == 0
            or
            (epoch + 1) % 100 == 0
    ):

        print(
            "Epoch:",
            epoch + 1,
            "| Loss:",
            loss.item()
        )


print()


# ==================================================
# 11. MODEL PREDICTIONS
# ==================================================

print("TEST 11: Predictions")
print()


with torch.no_grad():

    predictions = model(
        X_normalized
    )


for actual, predicted in zip(
        y,
        predictions
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
# 12. TRAINING ERROR
# ==================================================

print("TEST 12: Final Training Error")
print()


with torch.no_grad():

    final_loss = loss_function(
        predictions,
        y
    )


print(
    "Final MSE:",
    final_loss.item()
)

print()


# ==================================================
# 13. NEW MACHINE
# ==================================================

print("TEST 13: New Machine")
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
    "New machine:",
    new_machine
)

print()

print(
    "Predicted risk:",
    new_prediction.item()
)

print()


# ==================================================
# 14. MODEL PARAMETERS
# ==================================================

print("TEST 14: Learned Parameters")
print()


total_parameters = 0


for name, parameter in (
        model.named_parameters()
):

    parameter_count = (
        parameter.numel()
    )

    total_parameters += (
        parameter_count
    )

    print(
        name,
        "->",
        parameter.shape
    )

print()

print(
    "Total trainable parameters:",
    total_parameters
)

print()


# ==================================================
# 15. NEURAL NETWORK FLOW
# ==================================================

print("NEURAL NETWORK FLOW")
print()

print("Input features")
print("      ↓")
print("Linear layer")
print("      ↓")
print("ReLU")
print("      ↓")
print("Linear layer")
print("      ↓")
print("ReLU")
print("      ↓")
print("Output layer")
print("      ↓")
print("Prediction")

print()


# ==================================================
# 16. WHAT THE NETWORK LEARNS
# ==================================================

print("WHAT THE NETWORK LEARNS")
print()

print(
    "The network contains trainable weights "
    "and biases."
)

print()

print(
    "Training adjusts these parameters "
    "to reduce prediction error."
)

print()

print(
    "The optimizer updates the parameters "
    "using gradients from backpropagation."
)

print()


# ==================================================
# 17. MACHINE LEARNING VS DEEP LEARNING
# ==================================================

print("ML VS DEEP LEARNING")
print()

print(
    "Classical ML often relies on algorithms "
    "such as linear regression, decision trees, "
    "and random forests."
)

print()

print(
    "Deep learning uses neural networks with "
    "multiple layers of learned parameters."
)

print()


# ==================================================
# 18. FUTURE LLM CONNECTION
# ==================================================

print("FUTURE LLM CONNECTION")
print()

print(
    "Large language models are neural networks "
    "with very large numbers of parameters."
)

print()

print(
    "Modern language models use transformer "
    "architectures rather than this simple "
    "feed-forward network."
)

print()

print(
    "This lesson gives us the foundation for "
    "understanding those larger systems."
)

print()


# ==================================================
# 19. CURRENT SILVERWING PIPELINE
# ==================================================

print("CURRENT SILVERWING ML PIPELINE")
print()

print("Data")
print(" ↓")
print("Preprocessing")
print(" ↓")
print("Features")
print(" ↓")
print("Neural Network")
print(" ↓")
print("Training")
print(" ↓")
print("Backpropagation")
print(" ↓")
print("Prediction")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 31 COMPLETE ===")
