import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Generate ground truth using finite difference
def generate_ground_truth(Lx, Ly, nx, ny, nt, alpha, T_init, T_bc):
   dx = Lx / (nx - 1)
   dy = Ly / (ny - 1)
   dt = min(dx**2, dy**2) / (4 * alpha)  # CFL condition

   x = np.linspace(0, Lx, nx)
   y = np.linspace(0, Ly, ny)
   t = np.linspace(0, dt * nt, nt)
   X, Y = np.meshgrid(x, y)

   T = np.zeros((nt, ny, nx))
   T[0, :, :] = T_init(X, Y)

   for n in range(0, nt - 1):
       T[n + 1, 1:-1, 1:-1] = T[n, 1:-1, 1:-1] + alpha * dt * (
           (T[n, 2:, 1:-1] - 2 * T[n, 1:-1, 1:-1] + T[n, :-2, 1:-1]) / dx**2
           + (T[n, 1:-1, 2:] - 2 * T[n, 1:-1, 1:-1] + T[n, 1:-1, :-2]) / dy**2
       )

       # Apply boundary conditions
       T[n + 1, :, 0] = T_bc(X[:, 0], Y[:, 0])
       T[n + 1, :, -1] = T_bc(X[:, -1], Y[:, -1])
       T[n + 1, 0, :] = T_bc(X[0, :], Y[0, :])
       T[n + 1, -1, :] = T_bc(X[-1, :], Y[-1, :])

   return X, Y, t, T

def T_init(X, Y):
   return np.sin(np.pi * X) * np.sin(np.pi * Y)

def T_bc(X, Y):
   return 0


# Generate ground truth data
Lx, Ly = 1.0, 1.0
nx, ny, nt = 50, 50, 100
alpha = 0.01
X, Y, t, T_true = generate_ground_truth(Lx, Ly, nx, ny, nt, alpha, T_init, T_bc)

x_grid = X.flatten()[:, None]
y_grid = Y.flatten()[:, None]

t_grid = np.zeros((nx*ny*nt, 1))
x_repeated = np.zeros((nx*ny*nt, 1))
y_repeated = np.zeros((nx*ny*nt, 1))
T_flat = np.zeros((nx*ny*nt, 1))

for i in range(nt):
   t_grid[i*(nx*ny):(i+1)*(nx*ny), 0] = t[i]
   x_repeated[i*(nx*ny):(i+1)*(nx*ny), 0] = x_grid[:, 0]
   y_repeated[i*(nx*ny):(i+1)*(nx*ny), 0] = y_grid[:, 0]
   T_flat[i*(nx*ny):(i+1)*(nx*ny), 0] = T_true[i].flatten()
   
   
x_tensor = torch.tensor(x_repeated, dtype=torch.float32, requires_grad=True)
y_tensor = torch.tensor(y_repeated, dtype=torch.float32, requires_grad=True)
t_tensor = torch.tensor(t_grid, dtype=torch.float32, requires_grad=True)
T_tensor = torch.tensor(T_flat, dtype=torch.float32)

# Split data
n_samples = len(x_tensor)
train_idx = int(0.7 * n_samples)
val_idx = int(0.85 * n_samples)

x_train = x_tensor[:train_idx]
y_train = y_tensor[:train_idx]
t_train = t_tensor[:train_idx]
x_val = x_tensor[train_idx:val_idx]
y_val = y_tensor[train_idx:val_idx]
t_val = t_tensor[train_idx:val_idx]
x_test = x_tensor[val_idx:]
y_test = y_tensor[val_idx:]
t_test = t_tensor[val_idx:]

T_train = T_tensor[:train_idx]
T_val = T_tensor[train_idx:val_idx]
T_test = T_tensor[val_idx:]

class PINN(nn.Module):
   def __init__(self, layers):
       super(PINN, self).__init__()
       self.linears = nn.ModuleList()
       for i in range(len(layers) - 1):
           self.linears.append(nn.Linear(layers[i], layers[i + 1]))
       self.activation = nn.Tanh()

   def forward(self, x, y, t):
       inputs = torch.cat([x, y, t], dim=1)
       for i in range(len(self.linears) - 1):
           inputs = self.activation(self.linears[i](inputs))
       return self.linears[-1](inputs)

def physics_loss(model, x, y, t, alpha):
   u_pred = model(x, y, t)
   
   u_t = autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
   u_x = autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
   u_y = autograd.grad(u_pred, y, torch.ones_like(u_pred), create_graph=True)[0]
   u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
   u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

   residual = u_t - alpha * (u_xx + u_yy)
   return torch.mean(residual**2)

layers = [3, 50, 50, 50, 1]
model = PINN(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_losses = []
val_losses = []
epochs = 1000

for epoch in range(epochs):
   model.train()
   optimizer.zero_grad()

   u_pred_train = model(x_train, y_train, t_train)
   data_loss = torch.mean((u_pred_train - T_train)**2)
   phys_loss = physics_loss(model, x_train, y_train, t_train, alpha)
   total_loss = data_loss + phys_loss

   total_loss.backward()
   optimizer.step()

   train_losses.append(total_loss.item())

   model.eval()
   with torch.no_grad():
       u_pred_val = model(x_val, y_val, t_val)
       val_loss = torch.mean((u_pred_val - T_val)**2)
       val_losses.append(val_loss.item())

   if epoch % 100 == 0:
       print(f"Epoch {epoch}, Train Loss: {total_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

model.eval()
with torch.no_grad():
   u_pred_test = model(x_test, y_test, t_test)
   test_loss = torch.mean((u_pred_test - T_test)**2)
   print(f"Test Loss: {test_loss.item():.6f}")

# Plot 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.imshow(T_true[-1], extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="hot")
plt.colorbar(label="Temperature")

plt.subplot(1, 2, 2)
plt.title("PINN Prediction")
T_pred_final = u_pred_test[-nx*ny:].reshape(ny, nx).detach().numpy()
plt.imshow(T_pred_final, extent=[0, 1, 0, 1], origin="lower", aspect="auto", cmap="hot")
plt.colorbar(label="Temperature")

plt.tight_layout()
plt.show()   