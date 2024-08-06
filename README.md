# Boundary Element Method for 3D Laplace Equation

## Problem Formulatiuon

$$\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2} = 0 \quad \textrm{for} \quad (x, y, z) \in \Omega$$ 

with Dirichlet

$$\phi = f(x, y, z) \quad \textrm{for} \quad (x, y, z) \in \partial \Omega_1$$

and Neumann boundary conditions:

$$\frac{\partial \phi}{\partial n} = g(x, y, z) \quad \textrm{for} \quad (x, y, z) \in \partial \Omega_2$$

where $\Omega$ - three dimensional region with boundary $\partial \Omega = \partial \Omega_1 \cup \partial \Omega_2$, $f(x, y, z)$ and $g(x, y, z)$ - arbitrary functions. It is also worth noting that $n = (n_x, n_y, n_z)$ - unit normal vector to the surface $\partial \Omega$ and $\frac{\partial \phi}{\partial n} = n_x \frac{\partial \phi}{\partial x} + n_y \frac{\partial \phi}{\partial y} + n_z \frac{\partial \phi}{\partial z}$.

## Usage

The program is implemented in the form of a main class \textbf{BEMLaplace3D}, which contains the main computationally complex ones - such as solving a system of linear equations, and calculating a solution for a given point in space. This class was accelerated using the \textbf{Numba} library, which gave a significant boost. The code also contains some auxiliary subroutines for working with the geometry specified in the STL format file and some simple subroutines for postprocessing.
    
    Let's describe an example of working with code to solve a specific problem. First of all, you need to set the geometry of the surface that limits the computational area; for a STL file this is done using a simple function \textbf{stl2points} that returns the $(N, 3)$ arrays of $x$, $y$ and $z$ coordinates of the boundary elements (triangles) and their number - $N$. Next, an instance of the class \textbf{BEMLaplace3D} is initialized, which we will use to solve the equation:
    
    \begin{python}
    N, xt, yt, zt = stl2points(stl_file_path)
    solver = BEMLaplace3D(N, xt, yt, zt)
    \end{python}

    The next step is to set the boundary conditions. This is done using two $(N, )$ arrays bct and bcv (boundary condition tracker and boundary condition value), bct can contain either 0 - to indicate that Dirichlet boundary conditions are specified, or 1 - to indicate that Neumann boundary conditions are specified, bcv contains the corresponding value of the boundary condition. An example of specifying dirichlet conditions:

    \begin{python}
    bct = np.zeros(N)
    bcv = np.zeros(N)
    
    for i in range(N):
      bct[i] = 0
      bcv[i] = boundary_condition
    \end{python}

    Also, if the initial equation being solved is a Poisson equation with a constant right-hand side, and not Laplace, the procedure that was described in the introduction is used. This function is implemented as a class method and is used as follows:
    
    \begin{python}
    bct, bcv = solver.bc_correction(bct, bcv, rhs_const)
    \end{python}

    The most computationally difficult step - solving a system of linear equations (\ref{eq:29}) - is also implemented as a class method:
    
    \begin{python}
    phi, dphi = solver.linear_system(bct, bcv)
    \end{python}

    Once the value of the function and the normal derivative on the boundary have been found, you can easily find a solution at any point. In order to find a solution in the computational domain, you can use functions to define a mesh. Once the value of the function and the normal derivative on the boundary have been found, you can easily find a solution at any point. In order to find a solution in the computational domain, you can use the \textbf{intersect\_uniform\_mesh} function to specify a 2D triangular unstructured mesh - cutting the region along the $z$-axis, and also to specify a 3D unstructured mesh of tetrahedrons, you can use the \textbf{volume\_mesh} function. Both functions use only a STL file to create the mesh, and some parameters specifying resolution and restrictions. Let us show an example with the function \textbf{intersect\_uniform\_mesh}:

    \begin{python}
    x, y, triangles, cutplane = intersect_uniform_mesh(stl_file_path, shift, scale, x_min, x_max, y_min, y_max, resolution, z_slice)

    n = x.shape[0]
    bem_sol = np.zeros(x.shape)
    
    for i in range(n):
      bem_sol[i] = solver.solution_poisson(x[i], y[i], z_slice, rhs_const, phi, dphi)
    \end{python}

    It is also worth noting that the method produces large errors at the boundaries of the solution area. Perhaps this can be corrected by placing points in the volume further from the boundary and subsequently interpolating to the boundary at intermediate points. You can get rid of artifacts at the solution boundary using the function \textbf{artifact\_correction}.

To use these functions:

1. Ensure that all necessary dependencies, such as `numpy`, `numpy-stl`, `matplotlib`, `vedo`, `numba` are installed.
2. Import the functions into your IDE.
