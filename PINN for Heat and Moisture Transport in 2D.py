import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = 50, 50  # Number of grid points
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
dt = 0.01  # Time step
nt = 200  # Number of time steps

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Physical parameters
kappa = 0.01  # Thermal diffusivity
D = 0.01  # Moisture diffusivity
v = np.array([0.1, 0.1])  # Wind velocity (u, v)
Q_T = np.zeros((ny, nx))  # Heat source
Q_T[20:30, 20:30] = 10.0  # Localized heat source
Q_W = np.zeros((ny, nx))  # Moisture source
Q_W[10:15, 10:15] = 5.0  # Localized moisture source


# Initial conditions
T = np.zeros((ny, nx))  # Initial temperature
W = np.zeros((ny, nx))  # Initial moisture

# Finite difference loop
for n in range(nt):
    # Temperature updates
    T_new = T.copy()
    T_new[1:-1, 1:-1] = (
        T[1:-1, 1:-1]
        + kappa * dt * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2
            + (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        )
        - dt * v[0] * (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dx)
        - dt * v[1] * (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dy)
        + dt * Q_T[1:-1, 1:-1]
    )
    T = T_new.copy()

    # Moisture updates
    W_new = W.copy()
    W_new[1:-1, 1:-1] = (
        W[1:-1, 1:-1]
        + D * dt * (
            (W[2:, 1:-1] - 2 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dx**2
            + (W[1:-1, 2:] - 2 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dy**2
        )
        - dt * v[0] * (W[2:, 1:-1] - W[:-2, 1:-1]) / (2 * dx)
        - dt * v[1] * (W[1:-1, 2:] - W[1:-1, :-2]) / (2 * dy)
        + dt * Q_W[1:-1, 1:-1]
    )
    W = W_new.copy()
    
    
# Plot 
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, T, levels=20, cmap="hot")
plt.colorbar()
plt.title("Temperature (Finite Difference)")
plt.subplot(1, 2, 2)
plt.contourf(X, Y, W, levels=20, cmap="Blues")
plt.colorbar()
plt.title("Moisture (Finite Difference)")
plt.show()


# Generate ground truth with finite difference
def generate_ground_truth(q_T, q_W, Lx, Ly, nx, ny, nt, T_bc, W_bc):
    """
    Generate synthetic ground truth using finite difference for heat and moisture equations.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dt = min(dx**2 / 4, dy**2 / 4)  # CFL condition for stability
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize temperature (T) and moisture (W)
    T = np.zeros((ny, nx))
    W = np.zeros((ny, nx))

    # Source terms
    Q_T = np.zeros((ny, nx))
    Q_T[20:30, 20:30] = q_T
    Q_W = np.zeros((ny, nx))
    Q_W[10:15, 10:15] = q_W

    for n in range(nt):
        T_new = T.copy()
        W_new = W.copy()

        # Update T using finite difference
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2
            + (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
            + Q_T[1:-1, 1:-1]
        )

        # Update W using finite difference
        W_new[1:-1, 1:-1] = W[1:-1, 1:-1] + dt * (
            (W[2:, 1:-1] - 2 * W[1:-1, 1:-1] + W[:-2, 1:-1]) / dx**2
            + (W[1:-1, 2:] - 2 * W[1:-1, 1:-1] + W[1:-1, :-2]) / dy**2
            + Q_W[1:-1, 1:-1]
        )

        # Apply Dirichlet boundary conditions
        T_new[:, 0], T_new[:, -1], T_new[0, :], T_new[-1, :] = T_bc
        W_new[:, 0], W_new[:, -1], W_new[0, :], W_new[-1, :] = W_bc

        T, W = T_new, W_new

    return X, Y, T, W


# Generate synthetic ground truth
Lx, Ly = 1.0, 1.0
nx, ny = 50, 50
nt = 200
q_T, q_W = 1.0, 0.5
T_bc = [1.0, 0.0, 1.0, 0.0]
W_bc = [0.5, 0.0, 0.5, 0.0]

X, Y, T, W = generate_ground_truth(q_T, q_W, Lx, Ly, nx, ny, nt, T_bc, W_bc)

x_tensor = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).requires_grad_(True)
y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).requires_grad_(True)
T_tensor = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)
W_tensor = torch.tensor(W.flatten(), dtype=torch.float32).unsqueeze(1)


indices = np.arange(x_tensor.shape[0])
np.random.shuffle(indices)
train_split = int(0.7 * len(indices))
val_split = int(0.85 * len(indices))

train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
T_train, W_train = T_tensor[train_idx], W_tensor[train_idx]
x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]
T_val, W_val = T_tensor[val_idx], W_tensor[val_idx]
x_test, y_test = x_tensor[test_idx], y_tensor[test_idx]
T_test, W_test = T_tensor[test_idx], W_tensor[test_idx]



class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        for i in range(len(self.linears) - 1):
            xy = self.activation(self.linears[i](xy))
        return self.linears[-1](xy)


layers = [2, 50, 50, 50, 2]  # Input: (x, y) -> Outputs: (T, W)
model = PINN(layers)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def physics_loss(model, x, y):
    T_pred, W_pred = model(x, y).split(1, dim=1)

    # First derivatives
    T_x = autograd.grad(T_pred, x, torch.ones_like(T_pred), create_graph=True)[0]
    T_y = autograd.grad(T_pred, y, torch.ones_like(T_pred), create_graph=True)[0]
    W_x = autograd.grad(W_pred, x, torch.ones_like(W_pred), create_graph=True)[0]
    W_y = autograd.grad(W_pred, y, torch.ones_like(W_pred), create_graph=True)[0]

    # Second derivatives
    T_xx = autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_yy = autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]
    W_xx = autograd.grad(W_x, x, torch.ones_like(W_x), create_graph=True)[0]
    W_yy = autograd.grad(W_y, y, torch.ones_like(W_y), create_graph=True)[0]

    # Physics residuals
    T_residual = T_xx + T_yy - q_T
    W_residual = W_xx + W_yy - q_W

    return torch.mean(T_residual**2) + torch.mean(W_residual**2)


train_losses = []
val_losses = []
epochs = 10000

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    T_pred_train, W_pred_train = model(x_train, y_train).split(1, dim=1)
    data_loss_train = torch.mean((T_pred_train - T_train)**2) + torch.mean((W_pred_train - W_train)**2)
    
    phys_loss_train = physics_loss(model, x_train, y_train)

    total_loss_train = data_loss_train + phys_loss_train
    total_loss_train.backward(retain_graph=True)  
    optimizer.step()

    train_losses.append(total_loss_train.item())
    model.eval()
    x_val = x_val.clone().detach().requires_grad_(True)
    y_val = y_val.clone().detach().requires_grad_(True)

    with torch.no_grad():
        T_pred_val, W_pred_val = model(x_val, y_val).split(1, dim=1)
        data_loss_val = torch.mean((T_pred_val - T_val)**2) + torch.mean((W_pred_val - W_val)**2)

    phys_loss_val = physics_loss(model, x_val, y_val)
    total_loss_val = data_loss_val + phys_loss_val
    val_losses.append(total_loss_val.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Train Loss: {total_loss_train.item()}, Val Loss: {total_loss_val.item()}")
        

# Plot 
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
x_test = x_test.clone().detach().requires_grad_(True)
y_test = y_test.clone().detach().requires_grad_(True)
x_test = x_test.clone().detach().requires_grad_(True)
y_test = y_test.clone().detach().requires_grad_(True)

model.eval()

T_pred_test, W_pred_test = model(x_test, y_test).split(1, dim=1)
test_data_loss = torch.mean((T_pred_test - T_test)**2) + torch.mean((W_pred_test - W_test)**2)
test_phys_loss = physics_loss(model, x_test, y_test)

test_total_loss = test_data_loss + test_phys_loss
print(f"Test Data Loss: {test_data_loss.item()}, Test Physics Loss: {test_phys_loss.item()}, Test Total Loss: {test_total_loss.item()}")


# Visualize 
T_pred_test = T_pred_test.detach().numpy().reshape(-1)
W_pred_test = W_pred_test.detach().numpy().reshape(-1)
T_test = T_test.detach().numpy().reshape(-1)
W_test = W_test.detach().numpy().reshape(-1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(T_test, T_pred_test, alpha=0.7, label="Temperature")
plt.plot([T_test.min(), T_test.max()], [T_test.min(), T_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("True Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Test Set: Temperature")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(W_test, W_pred_test, alpha=0.7, label="Moisture")
plt.plot([W_test.min(), W_test.max()], [W_test.min(), W_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("True Moisture")
plt.ylabel("Predicted Moisture")
plt.title("Test Set: Moisture")
plt.legend()

plt.show()

from scipy.interpolate import griddata

nx, ny = 50, 50  
x_uniform = np.linspace(x_test.detach().numpy().min(), x_test.detach().numpy().max(), nx)
y_uniform = np.linspace(y_test.detach().numpy().min(), y_test.detach().numpy().max(), ny)
X_uniform, Y_uniform = np.meshgrid(x_uniform, y_uniform)


x_test_np = x_test.detach().numpy().flatten() if isinstance(x_test, torch.Tensor) else x_test.flatten()
y_test_np = y_test.detach().numpy().flatten() if isinstance(y_test, torch.Tensor) else y_test.flatten()
T_test_np = T_test.flatten()  
W_test_np = W_test.flatten()  
T_pred_test_np = T_pred_test.flatten()  
W_pred_test_np = W_pred_test.flatten()

# Interpolate the actual and predicted values onto the uniform grid
T_actual_grid = griddata(
    points=(x_test_np, y_test_np),
    values=T_test_np,
    xi=(X_uniform, Y_uniform),
    method='cubic'
)

T_predicted_grid = griddata(
    points=(x_test_np, y_test_np),
    values=T_pred_test_np,
    xi=(X_uniform, Y_uniform),
    method='cubic'
)

W_actual_grid = griddata(
    points=(x_test_np, y_test_np),
    values=W_test_np,
    xi=(X_uniform, Y_uniform),
    method='cubic'
)

W_predicted_grid = griddata(
    points=(x_test_np, y_test_np),
    values=W_pred_test_np,
    xi=(X_uniform, Y_uniform),
    method='cubic'
)


# Plot heatmaps for Temperature
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X_uniform, Y_uniform, T_actual_grid, levels=20, cmap="hot")
plt.colorbar(label="Temperature")
plt.title("Actual Temperature (Ground Truth)")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.contourf(X_uniform, Y_uniform, T_predicted_grid, levels=20, cmap="hot")
plt.colorbar(label="Temperature")
plt.title("Predicted Temperature (PINN)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# Plot heatmaps for Moisture
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X_uniform, Y_uniform, W_actual_grid, levels=20, cmap="Blues")
plt.colorbar(label="Moisture")
plt.title("Actual Moisture (Ground Truth)")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.contourf(X_uniform, Y_uniform, W_predicted_grid, levels=20, cmap="Blues")
plt.colorbar(label="Moisture")
plt.title("Predicted Moisture (PINN)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()


# Calculate residuals
T_residual = T_test - T_pred_test
W_residual = W_test - W_pred_test

residual_size = T_residual.shape[0]
train_split_residual = int(0.7 * residual_size)
val_split_residual = int(0.85 * residual_size)

residual_indices = np.arange(residual_size)
np.random.shuffle(residual_indices)

train_idx_residual = residual_indices[:train_split_residual]
val_idx_residual = residual_indices[train_split_residual:val_split_residual]
test_idx_residual = residual_indices[val_split_residual:]

T_residual_train, T_residual_val, T_residual_test = (
    T_residual[train_idx_residual],
    T_residual[val_idx_residual],
    T_residual[test_idx_residual],
)

W_residual_train, W_residual_val, W_residual_test = (
    W_residual[train_idx_residual],
    W_residual[val_idx_residual],
    W_residual[test_idx_residual],
)


# Residual PINN
model_residual = PINN(layers)
optimizer_residual = torch.optim.Adam(model_residual.parameters(), lr=0.001)

train_residual_losses = []
val_residual_losses = []

epochs_residual = 5000
for epoch in range(epochs_residual):
    model_residual.train()
    optimizer_residual.zero_grad()
    T_residual_pred_train, W_residual_pred_train = model_residual(x_test[train_idx_residual], y_test[train_idx_residual]).split(1, dim=1)
    loss_train_residual = (
        torch.mean((T_residual_pred_train - torch.tensor(T_residual_train).unsqueeze(1)) ** 2)
        + torch.mean((W_residual_pred_train - torch.tensor(W_residual_train).unsqueeze(1)) ** 2)
    )

    loss_train_residual.backward()
    optimizer_residual.step()
    train_residual_losses.append(loss_train_residual.item())
    model_residual.eval()
    with torch.no_grad():
        T_residual_pred_val, W_residual_pred_val = model_residual(
            x_test[val_idx_residual], y_test[val_idx_residual]
        ).split(1, dim=1)
        loss_val_residual = (
            torch.mean((T_residual_pred_val - torch.tensor(T_residual_val).unsqueeze(1)) ** 2)
            + torch.mean((W_residual_pred_val - torch.tensor(W_residual_val).unsqueeze(1)) ** 2)
        )
        val_residual_losses.append(loss_val_residual.item())

    if epoch % 500 == 0:
        print(f"Residual PINN Epoch {epoch}, Train Loss: {loss_train_residual.item()}, Val Loss: {loss_val_residual.item()}")


model_residual.eval()
with torch.no_grad():
    T_residual_pred_test, W_residual_pred_test = model_residual(
        x_test[test_idx_residual], y_test[test_idx_residual]
    ).split(1, dim=1)

# Combine original predictions with residual predictions
T_combined = T_pred_test[test_idx_residual] + T_residual_pred_test.numpy().reshape(-1)
W_combined = W_pred_test[test_idx_residual] + W_residual_pred_test.numpy().reshape(-1)

T_combined_error = np.mean((T_combined - T_test[test_idx_residual]) ** 2)
W_combined_error = np.mean((W_combined - W_test[test_idx_residual]) ** 2)

print(f"Combined Temperature Error: {T_combined_error}")
print(f"Combined Moisture Error: {W_combined_error}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_residual_losses, label="Training Loss (Residuals)")
plt.plot(val_residual_losses, label="Validation Loss (Residuals)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Residual PINN Training and Validation Loss")
plt.show()

# Scatter plots for the ensemble predictions
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(T_test[test_idx_residual], T_combined, alpha=0.7, label="Temperature (Ensemble)")
plt.plot(
    [T_test[test_idx_residual].min(), T_test[test_idx_residual].max()],
    [T_test[test_idx_residual].min(), T_test[test_idx_residual].max()],
    'r--',
    label="Ideal Fit",
)
plt.xlabel("True Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Ensemble Prediction: Temperature")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(W_test[test_idx_residual], W_combined, alpha=0.7, label="Moisture (Ensemble)")
plt.plot(
    [W_test[test_idx_residual].min(), W_test[test_idx_residual].max()],
    [W_test[test_idx_residual].min(), W_test[test_idx_residual].max()],
    'r--',
    label="Ideal Fit",
)
plt.xlabel("True Moisture")
plt.ylabel("Predicted Moisture")
plt.title("Ensemble Prediction: Moisture")
plt.legend()

plt.tight_layout()
plt.show()

T_residual_train = T_train - T_pred_train.reshape(-1)
W_residual_train = W_train - W_pred_train.reshape(-1)

T_residual_val = T_val - T_pred_val.reshape(-1)
W_residual_val = W_val - W_pred_val.reshape(-1)

T_residual_test = T_test - T_pred_test.reshape(-1)
W_residual_test = W_test - W_pred_test.reshape(-1)


x_test_np = x_test.detach().numpy().flatten()
y_test_np = y_test.detach().numpy().flatten()
T_residual_np = T_residual_test.flatten()  
W_residual_np = W_residual_test.flatten()  

nx, ny = 50, 50 
grid_x = np.linspace(x_test_np.min(), x_test_np.max(), nx)
grid_y = np.linspace(y_test_np.min(), y_test_np.max(), ny)
X_grid, Y_grid = np.meshgrid(grid_x, grid_y)

T_residual_grid = griddata(
    (x_test_np, y_test_np), T_residual_np, (X_grid, Y_grid), method="cubic"
)
W_residual_grid = griddata(
    (x_test_np, y_test_np), W_residual_np, (X_grid, Y_grid), method="cubic"
)

# Plot heatmaps of residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X_grid, Y_grid, T_residual_grid, levels=50, cmap="coolwarm")
plt.colorbar(label="Temperature Residual")
plt.title("Temperature Residual Heatmap")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
plt.contourf(X_grid, Y_grid, W_residual_grid, levels=50, cmap="coolwarm")
plt.colorbar(label="Moisture Residual")
plt.title("Moisture Residual Heatmap")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()


