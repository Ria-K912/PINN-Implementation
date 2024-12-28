import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

def generate_ground_truth(q, L, n_points, T_left, T_right):
    """
    Solve d²T/dx² = q using finite difference.
    Args:
        q (float): Heat source term.
        L (float): Length of the domain.
        n_points (int): Number of grid points.
        T_left (float): Boundary condition at x = 0.
        T_right (float): Boundary condition at x = L.

    Returns:
        x (numpy array): Grid points.
        T (numpy array): Temperature at grid points.
    """
    dx = L / (n_points - 1)  # Step Size
    x = np.linspace(0, L, n_points)

    # Coefficient matrix for finite difference
    A = np.zeros((n_points, n_points))
    b = np.full(n_points, -q)  

    # Internal nodes
    for i in range(1, n_points - 1):
        A[i, i - 1] = 1 / dx**2
        A[i, i] = -2 / dx**2
        A[i, i + 1] = 1 / dx**2

    # Boundary conditions
    A[0, 0] = 1
    b[0] = T_left
    A[-1, -1] = 1
    b[-1] = T_right

    T = np.linalg.solve(A, b)
    return x, T


# Parameters 
q = 1.0  # Heat source term
L = 1.0  # Length of the domain
n_points = 100  # Number of grid points
T_left = 1.0  # Temperature at x = 0
T_right = 0.0  # Temperature at x = L

x, T_exact = generate_ground_truth(q, L, n_points, T_left, T_right)
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
T_tensor = torch.tensor(T_exact, dtype=torch.float32).unsqueeze(1)


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()
    
    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        return self.linears[-1](x)
    

layers = [1, 20, 20, 20, 20, 1]  # Input -> 4 hidden layers -> Output
model = PINN(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def physics_loss(model, x):
    T_pred = model(x)
    T_x = autograd.grad(T_pred, x, torch.ones_like(T_pred), create_graph=True)[0]
    T_xx = autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    physics_residual = T_xx + q  # d²T/dx² + q = 0
    return torch.mean(physics_residual**2)



epochs = 5000
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    T_pred = model(x_tensor)
    data_loss = torch.mean((T_pred - T_tensor)**2)

    # Physics loss
    phys_loss = physics_loss(model, x_tensor)

    # Total loss
    total_loss = data_loss + phys_loss
    total_loss.backward()
    optimizer.step()

    losses.append(total_loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Data Loss: {data_loss.item():.5f}, Physics Loss: {phys_loss.item():.5f}, Total Loss: {total_loss.item():.5f}")

# Plot 
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Total Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("PINN Training Loss")
plt.legend()
plt.show()

# Evaluate
model.eval()
with torch.no_grad():
    x_test = np.linspace(0, L, n_points).reshape(-1, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    T_pred = model(x_test_tensor).detach().numpy()

# Plot 
plt.figure(figsize=(8, 5))
plt.plot(x, T_exact, label="Exact Solution (Finite Difference)", color="blue", linewidth=2)
plt.plot(x_test, T_pred, label="PINN Prediction", linestyle="dashed", color="red", linewidth=2)
plt.xlabel("x")
plt.ylabel("Temperature T(x)")
plt.title("Exact Solution vs PINN Prediction")
plt.legend()
plt.show()



