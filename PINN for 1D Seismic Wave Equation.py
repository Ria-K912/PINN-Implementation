import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i in range(len(self.linears) - 1):
            inputs = self.activation(self.linears[i](inputs))
        return self.linears[-1](inputs)

def seismic_wave_loss(model, x, t, c):
    u_pred = model(x, t)
    u_t = autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    u_tt = autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]

    u_x = autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_tt - c**2 * u_xx
    return torch.mean(residual**2)

def boundary_initial_conditions_loss(model, x, t, c):
    # Initial condition: u(x, 0) = sin(pi * x), du/dt(x, 0) = 0
    t0 = torch.zeros_like(x, requires_grad=True)
    u0_pred = model(x, t0)
    u0_true = torch.sin(np.pi * x)
    u_t0_pred = autograd.grad(u0_pred, t0, torch.ones_like(u0_pred), create_graph=True, allow_unused=True)[0]

    initial_loss = torch.mean((u0_pred - u0_true)**2) + torch.mean(u_t0_pred**2)

    # Boundary conditions: u(0, t) = u(1, t) = 0
    x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32, requires_grad=True).repeat(t.shape[0], 1)
    t_boundary = t.repeat(2, 1).T.reshape(-1, 1) 
    u_boundary_pred = model(x_boundary, t_boundary)
    boundary_loss = torch.mean(u_boundary_pred**2)

    return initial_loss + boundary_loss


# Finite Difference Method (FDM) for ground truth
def finite_difference_wave(nx, nt, L, T, c):
    dx = L / (nx - 1)
    dt = T / (nt - 1)

    # CFL condition
    CFL = c * dt / dx
    if CFL > 1:
        raise ValueError("CFL condition not satisfied. Reduce dt or increase dx.")
    u = np.zeros((nt, nx))

    # Initial conditions
    x = np.linspace(0, L, nx)
    u[0, :] = np.sin(np.pi * x)  # u(x, 0) = sin(pi * x)
    u[1, :] = u[0, :]  # du/dt(x, 0) = 0
    
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[n + 1, i] = (
                2 * u[n, i] - u[n - 1, i]
                + CFL**2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
            )

    return u


L = 1.0  # Domain length
T = 1.0  # Total time
nx, nt = 100, 100
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
X, T_grid = np.meshgrid(x, t)

x_tensor = torch.tensor(X.flatten(), dtype=torch.float32, requires_grad=True).unsqueeze(1)
t_tensor = torch.tensor(T_grid.flatten(), dtype=torch.float32, requires_grad=True).unsqueeze(1)


c = 1.0  # Wave speed for seismic equation
layers = [2, 50, 50, 50, 1]  # 2 inputs: x, t -> Output: u(x, t)
model = PINN(layers)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5000
train_losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    physics_loss = seismic_wave_loss(model, x_tensor, t_tensor, c)
    boundary_loss = boundary_initial_conditions_loss(model, x_tensor, t_tensor, c)
    total_loss = physics_loss + boundary_loss
    total_loss.backward()
    optimizer.step()
    train_losses.append(total_loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item()}")
model.eval()
with torch.no_grad():
    u_pred_test = model(x_tensor, t_tensor).detach().numpy()

u_pred_test_grid = u_pred_test.reshape(nt, nx)
u_fdm = finite_difference_wave(nx, nt, L, T, c)



plt.figure(figsize=(15, 6))

# FDM Solution
plt.subplot(1, 2, 1)
plt.imshow(
    u_fdm,
    extent=[0, L, 0, T],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.colorbar(label="Displacement")
plt.title("FDM Solution (Seismic Wave)")

# PINN solution plot
plt.subplot(1, 2, 2)
plt.imshow(
    u_pred_test_grid,
    extent=[0, L, 0, T],
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
plt.colorbar(label="Displacement")
plt.title("PINN Solution (Seismic Wave)")

plt.tight_layout()
plt.show()

# Scatter Plot: FDM vs PINN
plt.figure(figsize=(8, 8))
plt.scatter(u_fdm.flatten(), u_pred_test.flatten(), alpha=0.5)
plt.xlabel("FDM Solution")
plt.ylabel("PINN Prediction")
plt.title("Comparison of FDM vs PINN Prediction")
plt.plot([u_fdm.min(), u_fdm.max()], [u_fdm.min(), u_fdm.max()], 'r--')  # y=x line
plt.grid()
plt.show()
