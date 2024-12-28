# Physics-Informed Neural Networks (PINNs) and Finite Difference Methods (FDM)

This repository contains implementations of Physics-Informed Neural Networks (PINNs) and Finite Difference Methods (FDM) for solving various physics-based equations. The codes showcase the integration of neural networks with physical laws to solve partial differential equations (PDEs) and compare their results with traditional numerical methods.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Code Files](#code-files)
3. [Dependencies](#dependencies)
4. [Usage](#usage)
5. [Results and Visualization](#results-and-visualization)
6. [References](#references)

---

## Introduction
Physics-Informed Neural Networks (PINNs) are a powerful tool for solving PDEs by leveraging neural networks and enforcing physical constraints. This repository includes:
- PINNs for 1D and 2D heat equations.
- PINNs for 1D seismic wave equations.
- Comparison with Finite Difference Methods (FDM) for validation.
- PINNs for experimental data prediction.

---

## Code Files
Below is a summary of the included codes and their purposes:

### Code 2 - PINN for 1D Heat Equation:
- Solves the equation:
  $$
  \frac{\partial^2 T}{\partial x^2} = q
  $$
  using PINNs.
- Compares results with FDM-generated ground truth.

---

### Code 3 - PINN for Heat and Moisture Transport in 2D:
- Solves coupled heat and moisture diffusion equations:
  $$
  \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = q_T
  $$
  $$
  \frac{\partial^2 W}{\partial x^2} + \frac{\partial^2 W}{\partial y^2} = q_W
  $$
- Compares PINN predictions with FDM results.

---

### Code 4 - PINN for 2D Transient Heat Equation:
- Solves:
  $$
  \frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)
  $$
- Visualizes transient heat distribution.

---

### Code 5 - PINN for 1D Seismic Wave Equation:
- Solves:
  $$
  \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
  $$
- Validates results against FDM-generated ground truth.

---

## Dependencies
The following libraries are required to run the codes:
- **Python (>= 3.8)**
- **NumPy**
- **PyTorch**
- **Matplotlib**
- **Pandas** 
- **scikit-learn** 

To install dependencies, run:
```bash
pip install numpy torch matplotlib pandas scikit-learn
