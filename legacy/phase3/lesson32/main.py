
# Silverwing ML
# Phase 3 - Lesson 32
# Tensors, Gradients and Backpropagation


import torch
import torch.nn as nn
import torch.optim as optim


print("=== SILVERWING ML ===")
print("Phase 3 - Lesson 32")
print("Tensors, Gradients and Backpropagation")
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
# 2. CREATE A TENSOR
# ==================================================

print("TEST 2: Tensor")
print()

x = torch.tensor(
    [2.0, 4.0, 6.0],
    requires_grad=True
)

print("Tensor:")
print(x)

print()

print("Requires gradients:")
print(x.requires_grad)

print()


# ==================================================
# 3. TENSOR OPERATION
# ==================================================

print("TEST 3: Tensor Operation")
print()

y = x ** 2

print("x:")
print(x)

print()

print("y = x²:")
print(y)

print()


# ==================================================
# 4. SUM THE RESULTS
# ==================================================

print("TEST 4: Create Scalar Output")
print()

output = y.sum()

print("Output:")
print(output)

print()


# ==================================================
# 5. CALCULATE GRADIENTS
# ==================================================

print("TEST 5: Automatic Differentiation")
print()

output.backward()

print("Gradient of x:")
print(x.grad)

print()


# ==================================================
# 6. VERIFY THE GRADIENT
# ==================================================

print("TEST 6: Gradient Verification")
print()

print(
    "For y = x², the derivative is 2x."
)

print()

expected_gradient = 2 * x.detach()

print(
    "Expected gradient:"
)

print(
    expected_gradient
)

print()

print(
    "PyTorch gradient:"
)

print(
    x.grad
)

print()


# ==================================================
# 7. SIMPLE LINEAR MODEL
# ==================================================

print("TEST 7: Linear Model")
print()


model = nn.Linear(
    1,
    1
)


print(model)

print()


# ==================================================
# 8. TRAINING DATA
# ==================================================

print("TEST 8: Training Data")
print()

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])


y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0]
])


print("X:")
print(X)

print()

print("y:")
print(y)

print()


# ==================================================
# 9. MODEL BEFORE TRAINING
# ==================================================

print("TEST 9: Initial Prediction")
print()

with torch.no_grad():

    initial_prediction = model(X)


print(
    "Initial predictions:"
)

print(
    initial_prediction
)

print()


# ==================================================
# 10. LOSS FUNCTION
# ==================================================

print("TEST 10: Loss Function")
print()

loss_function = nn.MSELoss()

print(
    "Loss function:",
    type(loss_function).__name__
)

print()


# ==================================================
# 11. OPTIMIZER
# ==================================================

print("TEST 11: Optimizer")
print()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)


print(
    "Optimizer:",
    type(optimizer).__name__
)

print()


# ==================================================
# 12. TRAINING LOOP
# ==================================================

print("TEST 12: Training")
print()


epochs = 1000


for epoch in range(epochs):

    # ----------------------------------------------
    # Forward pass
    # ----------------------------------------------

    predictions = model(X)


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
    # Update parameters
    # ----------------------------------------------

    optimizer.step()


    # ----------------------------------------------
    # Display progress
    # ----------------------------------------------

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
# 13. TRAINED PREDICTIONS
# ==================================================

print("TEST 13: Trained Predictions")
print()

with torch.no_grad():

    predictions = model(X)


print(
    "Predictions:"
)

print(
    predictions
)

print()


# ==================================================
# 14. FINAL LOSS
# ==================================================

print("TEST 14: Final Loss")
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
# 15. LEARNED WEIGHT AND BIAS
# ==================================================

print("TEST 15: Learned Parameters")
print()


weight = model.weight.item()
bias = model.bias.item()


print(
    "Learned weight:",
    weight
)

print(
    "Learned bias:",
    bias
)

print()

print(
    "The model approximates:"
)

print(
    "y =",
    round(weight, 4),
    "* x +",
    round(bias, 4)
)

print()


# ==================================================
# 16. PREDICT A NEW VALUE
# ==================================================

print("TEST 16: New Prediction")
print()


new_x = torch.tensor([
    [10.0]
])


with torch.no_grad():

    new_prediction = model(
        new_x
    )


print(
    "Input:",
    new_x.item()
)

print(
    "Predicted output:",
    new_prediction.item()
)

print()


# ==================================================
# 17. INSPECT GRADIENTS
# ==================================================

print("TEST 17: Model Gradients")
print()


for name, parameter in model.named_parameters():

    print(
        "Parameter:",
        name
    )

    print(
        "Value:",
        parameter.detach()
    )

    print(
        "Gradient:",
        parameter.grad
    )

    print()


# ==================================================
# 18. MANUAL LEARNING CONCEPT
# ==================================================

print("TEST 18: Learning Concept")
print()

print("1. Start with parameters.")
print("2. Make a prediction.")
print("3. Calculate the loss.")
print("4. Calculate gradients.")
print("5. Update parameters.")
print("6. Repeat.")

print()


# ==================================================
# 19. WHY GRADIENTS MATTER
# ==================================================

print("WHY GRADIENTS MATTER")
print()

print(
    "A gradient tells the optimizer how "
    "a parameter affects the loss."
)

print()

print(
    "The optimizer uses this information "
    "to move parameters toward lower loss."
)

print()


# ==================================================
# 20. BACKPROPAGATION CONCEPT
# ==================================================

print("BACKPROPAGATION")
print()

print(
    "Backpropagation calculates how much "
    "each trainable parameter contributed "
    "to the prediction error."
)

print()

print(
    "Those gradients are then used by the "
    "optimizer to update the parameters."
)

print()


# ==================================================
# 21. NEURAL NETWORK LEARNING LOOP
# ==================================================

print("NEURAL NETWORK LEARNING LOOP")
print()

print("Input")
print(" ↓")
print("Forward pass")
print(" ↓")
print("Prediction")
print(" ↓")
print("Loss")
print(" ↓")
print("Backward pass")
print(" ↓")
print("Gradients")
print(" ↓")
print("Optimizer")
print(" ↓")
print("Updated parameters")
print(" ↓")
print("Repeat")

print()


# ==================================================
# 22. CONNECTION TO DEEP LEARNING
# ==================================================

print("DEEP LEARNING CONNECTION")
print()

print(
    "Large neural networks perform the same "
    "basic learning process across many layers "
    "and very large numbers of parameters."
)

print()

print(
    "The mathematics becomes much larger, "
    "but the fundamental learning loop remains."
)

print()


# ==================================================
# 23. CONNECTION TO LLMS
# ==================================================

print("LLM CONNECTION")
print()

print(
    "Large language models also learn by "
    "optimizing neural-network parameters "
    "using gradients."
)

print()

print(
    "Transformers introduce specialized "
    "architectures and attention mechanisms, "
    "which we will study later."
)

print()


# ==================================================
# 24. SILVERWING PIPELINE
# ==================================================

print("CURRENT SILVERWING LEARNING PIPELINE")
print()

print("Data")
print(" ↓")
print("Tensor")
print(" ↓")
print("Neural Network")
print(" ↓")
print("Forward Pass")
print(" ↓")
print("Loss")
print(" ↓")
print("Backpropagation")
print(" ↓")
print("Optimizer")
print(" ↓")
print("Updated Model")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 32 COMPLETE ===")