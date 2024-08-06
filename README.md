# Boundary Element Method for 3D Laplace Equation

## Overview

$$
 \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2} = 0 \quad \textrm{for} \quad (x, y, z) \in \Omega
$$
with boundary conditions:
$$
 \phi &= f(x, y, z) \quad \textrm{for} \quad (x, y, z) \in \partial \Omega_1
$$
$$
 \frac{\partial \phi}{\partial n} &= g(x, y, z) \quad \textrm{for} \quad (x, y, z) \in \partial \Omega_2
$$

where $\Omega$ - three dimensional region with boundary $\partial \Omega = \partial \Omega_1 \cup \partial \Omega_2$, $f(x, y, z)$ and $g(x, y, z)$ - arbitrary functions. It is also worth noting that $n = (n_x, n_y, n_z)$ - unit normal vector to the surface $\partial \Omega$ and $\frac{\partial \phi}{\partial n} = n_x \frac{\partial \phi}{\partial x} + n_y \frac{\partial \phi}{\partial y} + n_z \frac{\partial \phi}{\partial z}$.

## Usage

To use these functions:

1. Ensure that all necessary dependencies, such as `numpy`, `numpy-stl`, `matplotlib`, `vedo`, `numba` are installed.
2. Import the functions into your IDE.
