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

### Code 1 - PINN_Full_Code:
   - Consolidated version of all individual code files.
   - Combines PINN implementations for various problems, integrating comparisons with FDM.


### Code 2 - PINN for 1D Heat Equation:
- Solves the equation:![image](https://github.com/user-attachments/assets/c11dfd3c-b852-4840-a637-a3c06282ba02)  using PINNs.
---

### Code 3 - PINN for Heat and Moisture Transport in 2D:
- Solves coupled heat and moisture diffusion equations:![image](https://github.com/user-attachments/assets/3e220ef7-7038-406e-8096-3b559b459876)
---

### Code 4 - PINN for 2D Transient Heat Equation:
- Solves the equation:![image](https://github.com/user-attachments/assets/d9bfd752-2a82-496c-a4ed-c7e497be7165)
---

### Code 5 - PINN for 1D Seismic Wave Equation:
- Solves the equation:![image](https://github.com/user-attachments/assets/3793e212-0b30-4a4d-ac8c-84a23fdcbcaf)
  
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
