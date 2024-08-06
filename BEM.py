import numpy as np
from numba.experimental import jitclass
from numba import njit, int32, float64
import stl


@njit
def fund_sol3d(x, y, z, ksi, eta, zeta):
    """
    Function that calculates fundamental solution of Laplace 3D equation.
    :param x: float
    The x - coordinate of the point at which the fundamental solution is calculated.
    :param y: float
    The y - coordinate of the point at which the fundamental solution is calculated.
    :param z: float
    The z - coordinate of the point at which the fundamental solution is calculated.
    :param ksi: float
    Singular point x - coordinate of fundamental solution.
    :param eta: float
    Singular point y - coordinate of fundamental solution.
    :param zeta: float
    Singular point z - coordinate of fundamental solution.
    :return: float
    Value of the fundamental solution at point (x, y, z).
    """
    return -1 / (4 * np.pi) * 1 / np.sqrt((x - ksi) ** 2 + (y - eta) ** 2 + (z - zeta) ** 2)


@njit
def gauss_quad_points():
    """
    Initialization of points for 2D gaussian quadrature on a square domain [0, 1] x [0, 1].
    :return: array_like
    t_gauss, v_gauss - points coordinates of 2D gaussian quadrature.
    """
    n_points = 4
    n_blocks = 4

    t_gauss = np.zeros(n_points * n_blocks, dtype=np.float64)
    v_gauss = np.zeros(n_points * n_blocks, dtype=np.float64)

    sample_t = 1 / np.sqrt(3) * np.array([1, 1, -1, -1])
    sample_v = 1 / np.sqrt(3) * np.array([1, -1, 1, -1])

    size = int(np.sqrt(n_blocks))
    block = np.arange(0, n_blocks).reshape(size, size)

    for v_block in range(size):
        for t_block in range(size):
            block_id = n_points * block[v_block, t_block]
            t_gauss[block_id:block_id + n_points] = (sample_t + 1 + 2 * t_block) / 4
            v_gauss[block_id:block_id + n_points] = (sample_v + 1 + 2 * v_block) / 4

    return t_gauss, v_gauss


def stl2points(path):
    """
    Function for reading a stl file and obtaining the coordinates of the vertices of the boundary elements.
    :param path: string
    The path to the stl file.
    :return: tuple of int - N and array_like xt, yt, zt (N, 3), where N - number of boundary elements.
    Number of triangles and its vertices coordinates.
    """
    scaffold = stl.mesh.Mesh.from_file(path)
    vertices = scaffold.vectors
    N = vertices.shape[0]
    xt = np.array(vertices[:, :, 0], dtype=np.float64)
    yt = np.array(vertices[:, :, 1], dtype=np.float64)
    zt = np.array(vertices[:, :, 2], dtype=np.float64)
    return N, xt, yt, zt

@njit
def dfund_sol3d(x, y, z, ksi, eta, zeta, n_x, n_y, n_z):
    """
    Function that calculates derivative of fundamental solution along normal vector of Laplace 3D equation.
    :param x: float
    The x - coordinate of the point at which the fundamental solution is calculated.
    :param y: float
    The y - coordinate of the point at which the fundamental solution is calculated.
    :param z: float
    The z - coordinate of the point at which the fundamental solution is calculated.
    :param ksi: float
    Singular point x - coordinate of fundamental solution.
    :param eta: float
    Singular point y - coordinate of fundamental solution.
    :param zeta: float
    Singular point z - coordinate of fundamental solution.
    :param n_x: float
    The x - component of normal vector to boundary element.
    :param n_y: float
    The y - component of normal vector to boundary element.
    :param n_z: float
    The z - component of normal vector to boundary element.
    :return: float
    Value of the derivative of fundamental solution along normal vector (n_x, n_y, n_z) at point (x, y, z).
    """
    return ((x - ksi) * n_x + (y - eta) * n_y + (z - zeta) * n_z) / (4 * np.pi * (np.sqrt((x - ksi) ** 2 + (y - eta) ** 2 + (z - zeta) ** 2)) ** 3)


spec = [('N', int32), ('xt', float64[:, :]), ('yt', float64[:, :]), ('zt', float64[:, :]),
        ('n_x', float64[:]), ('n_y', float64[:]), ('n_z', float64[:]), ('jac', float64[:]),
        ('pc', float64[:, :, :]), ('xm', float64[:]), ('ym', float64[:]), ('zm', float64[:]),
        ('t_gauss', float64[:]), ('v_gauss', float64[:])]


@jitclass(spec)
class BEMLaplace3D:

    def __init__(self, N, xt, yt, zt):
        """
        Instance initialization of Boundary Element Method solver for 3D Laplace equation.
        Calculation of auxiliary elements: coordinates of the normal vector to the boundary element,
        setting points for Gaussian quadrature, area of the boundary element and its parameterization.
        :param N: int
        Number of boundary elements.
        :param xt: array_like (N, 3)
        The x - coordinates of triangle boundary element vertices.
        :param yt: array_like (N, 3)
        The y - coordinates of triangle boundary element vertices.
        :param zt: array_like (N, 3)
        The z - coordinates of triangle boundary element vertices.
        """
        self.N = N
        self.xt = xt
        self.yt = yt
        self.zt = zt

        x21 = xt[:, 1] - xt[:, 0]
        x31 = xt[:, 2] - xt[:, 0]
        x23 = xt[:, 1] - xt[:, 2]
        y21 = yt[:, 1] - yt[:, 0]
        y31 = yt[:, 2] - yt[:, 0]
        y23 = yt[:, 1] - yt[:, 2]
        z21 = zt[:, 1] - zt[:, 0]
        z31 = zt[:, 2] - zt[:, 0]
        z23 = zt[:, 1] - zt[:, 2]

        n_x = y21 * z31 - z21 * y31
        n_y = z21 * x31 - x21 * z31
        n_z = x21 * y31 - y21 * x31
        n_norm = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
        n_x = n_x / n_norm
        n_y = n_y / n_norm
        n_z = n_z / n_norm

        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        alpha = np.sqrt(x21 ** 2 + y21 ** 2 + z21 ** 2)
        beta = np.sqrt(x23 ** 2 + y23 ** 2 + z23 ** 2)
        gamma = np.sqrt(x31 ** 2 + y31 ** 2 + z31 ** 2)
        sigma = (alpha + beta + gamma) / 2
        jac = 2 * np.sqrt(sigma * (sigma - alpha) * (sigma - beta) * (sigma - gamma))

        self.jac = jac

        pc = np.zeros((N, 3, 3), dtype=np.float64)
        xm = np.zeros(N, dtype=np.float64)
        ym = np.zeros(N, dtype=np.float64)
        zm = np.zeros(N, dtype=np.float64)

        for i in range(N):

            if np.absolute(n_z[i]) >= 1 / np.sqrt(3):
                pc[i, 0, 0] = x21[i]
                pc[i, 0, 1] = x31[i]
                pc[i, 0, 2] = xt[i, 0]
                pc[i, 1, 0] = y21[i]
                pc[i, 1, 1] = y31[i]
                pc[i, 1, 2] = yt[i, 0]
                pc[i, 2, 0] = - (n_x[i] * pc[i, 0, 0] + n_y[i] * pc[i, 1, 0]) / n_z[i]
                pc[i, 2, 1] = - (n_x[i] * pc[i, 0, 1] + n_y[i] * pc[i, 1, 1]) / n_z[i]
                pc[i, 2, 2] = zt[i, 0]

            elif np.absolute(n_y[i]) >= 1 / np.sqrt(3):
                pc[i, 0, 0] = x21[i]
                pc[i, 0, 1] = x31[i]
                pc[i, 0, 2] = xt[i, 0]
                pc[i, 2, 0] = z21[i]
                pc[i, 2, 1] = z31[i]
                pc[i, 2, 2] = zt[i, 0]
                pc[i, 1, 0] = - (n_x[i] * pc[i, 0, 0] + n_z[i] * pc[i, 2, 0]) / n_y[i]
                pc[i, 1, 1] = - (n_x[i] * pc[i, 0, 1] + n_z[i] * pc[i, 2, 1]) / n_y[i]
                pc[i, 1, 2] = yt[i, 0]

            else:
                pc[i, 1, 0] = y21[i]
                pc[i, 1, 1] = y31[i]
                pc[i, 1, 2] = yt[i, 0]
                pc[i, 2, 0] = z21[i]
                pc[i, 2, 1] = z31[i]
                pc[i, 2, 2] = zt[i, 0]
                pc[i, 0, 0] = - (n_z[i] * pc[i, 2, 0] + n_y[i] * pc[i, 1, 0]) / n_x[i]
                pc[i, 0, 1] = - (n_z[i] * pc[i, 2, 1] + n_y[i] * pc[i, 1, 1]) / n_x[i]
                pc[i, 0, 2] = xt[i, 0]

            xm[i] = pc[i, 0, 0] / 4 + pc[i, 0, 1] / 2 + pc[i, 0, 2]
            ym[i] = pc[i, 1, 0] / 4 + pc[i, 1, 1] / 2 + pc[i, 1, 2]
            zm[i] = pc[i, 2, 0] / 4 + pc[i, 2, 1] / 2 + pc[i, 2, 2]

        self.pc = pc
        self.xm = xm
        self.ym = ym
        self.zm = zm

        t_gauss, v_gauss = gauss_quad_points(4, 4)

        self.t_gauss = t_gauss
        self.v_gauss = v_gauss

    def calculate_d(self, ksi, eta, zeta, k, nx, ny, nz):

        """
        Method for calculating double integrals of fundamental solution and it derivative using Gaussian quadrature.
        :param ksi: float
        Singular point x - coordinate of fundamental solution.
        :param eta: float
        Singular point y - coordinate of fundamental solution.
        :param zeta: float
        Singular point z - coordinate of fundamental solution.
        :param k: int
        Number of the boundary element on the surface of which the integral is calculated.
        :param nx: float
        The x - component of normal vector to boundary element.
        :param ny: float
        The y - component of normal vector to boundary element.
        :param nz: float
        The z - component of normal vector to boundary element.
        :return: d1, d2  - (float, float)
        A tuple containing the values of two integrals
        of the fundamental solution and it derivative along the normal vector.
        """

        jac = self.jac
        pc = self.pc
        t_gauss = self.t_gauss
        v_gauss = self.v_gauss

        d1 = 0
        d2 = 0

        for i in range(16):
            u_t = t_gauss[i] * (1 - v_gauss[i])
            x_k = pc[k, 0, 0] * u_t + pc[k, 0, 1] * v_gauss[i] + pc[k, 0, 2]
            y_k = pc[k, 1, 0] * u_t + pc[k, 1, 1] * v_gauss[i] + pc[k, 1, 2]
            z_k = pc[k, 2, 0] * u_t + pc[k, 2, 1] * v_gauss[i] + pc[k, 2, 2]
            d1 += fund_sol3d(x_k, y_k, z_k, ksi, eta, zeta) * (1 - v_gauss[i])
            d2 += dfund_sol3d(x_k, y_k, z_k, ksi, eta, zeta, nx, ny, nz) * (1 - v_gauss[i])

        d1 = jac[k] * d1 / 16
        d2 = jac[k] * d2 / 16

        return d1, d2

    def bc_correction(self, bct, bcv, c):
        """
        Method for correction of boundary conditions for solution of Poisson equation as sum of particular solution and
        solution of Laplace equation with new boundary conditions.
        :param bct: array_like (N) - number of boundary elements.
        Boundary condition tracker. Contain zeros and ones: 1 - if specified Neumann and
        0 - if specified Dirichlet boundary condition.
        :param bcv: array_like (N) - number of boundary elements.
        Array that contain boundary condition values.
        :param c: float
        Constant in right hand side of Poisson equation.
        :return: array_like
        New boundary conditions for Laplace equation.
        """

        N = self.N
        xm = self.xm
        ym = self.ym
        zm = self.zm
        n_x = self.n_x
        n_y = self.n_y
        n_z = self.n_z

        for i in range(N):

            if bct[i] == 0:
                bcv[i] = bcv[i] - c * (xm[i] ** 2 + ym[i] ** 2 + zm[i] ** 2) / 6
            else:
                bcv[i] = bcv[i] - c * (xm[i] * n_x[i] + ym[i] * n_y[i] + zm[i] * n_z[i]) / 3

        return bct, bcv

    def linear_system(self, bct, bcv):
        """
        A function that composes a system of linear equations and solves it to find the missing boundary conditions
        of both Dirichlet and Neumann.
        :param bct: array_like (N) - number of boundary elements.
        Boundary condition tracker. Contain zeros and ones: 1 - if specified Neumann and
        0 - if specified Dirichlet boundary condition.
        :param bcv: array_like (N) - number of boundary elements.
        Array that contain boundary condition values.
        :return: phi, dphi - array_like (N)
        Arrays that contain function and it derivative at boundary domain.
        """

        N = self.N

        nx = self.n_x
        ny = self.n_y
        nz = self.n_z

        xm = self.xm
        ym = self.ym
        zm = self.zm

        a = np.zeros((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)
        phi = np.zeros(N, dtype=np.float64)
        dphi = np.zeros(N, dtype=np.float64)

        for m in range(N):
            for k in range(N):

                d1, d2 = self.calculate_d(xm[m], ym[m], zm[m], k, nx[k], ny[k], nz[k])

                if k == m:
                    delta = 1
                else:
                    delta = 0

                if bct[k] == 0:
                    a[m, k] = -d1
                    b[m] += bcv[k] * (0.5 * delta - d2)
                else:
                    a[m, k] = d2 - 0.5 * delta
                    b[m] += bcv[k] * d1

        z = np.linalg.solve(a, b)

        for i in range(N):
            if bct[i] == 0:
                phi[i] = bcv[i]
                dphi[i] = z[i]
            else:
                phi[i] = z[i]
                dphi[i] = bcv[i]

        return phi, dphi

    def solution_laplace(self, ksi, eta, zeta, phi, dphi):
        """
        Method for calculating the solution of Laplace equation.
        :param ksi: float
        The x - coordinate at which the solution is calculated.
        :param eta: float
        The y - coordinate at which the solution is calculated.
        :param zeta: float
        The z - coordinate at which the solution is calculated.
        :param phi: array_like
        Function value on the boundary elements.
        :param dphi: array_like
        Function normal derivative value on the boundary elements.
        :return: float
        Solution of Laplace equation at a point (ksi, eta, zeta).
        """

        N = self.N
        nx = self.n_x
        ny = self.n_y
        nz = self.n_z

        sol = 0
        for i in range(N):
            d1, d2 = self.calculate_d(ksi, eta, zeta, i, nx[i], ny[i], nz[i])
            sol = sol + phi[i] * d2 - dphi[i] * d1

        return sol

    def solution_poisson(self, ksi, eta, zeta, c, phi, dphi):
        """
        Method for calculating the solution of Poisson equation.
        :param ksi: float
        The x - coordinate at which the solution is calculated.
        :param eta: float
        The y - coordinate at which the solution is calculated.
        :param zeta: float
        The z - coordinate at which the solution is calculated.
        :param c: float
        Constant in right hand side of Poisson equation.
        :param phi: array_like
        Function value on the boundary elements.
        :param dphi: array_like
        Function normal derivative value on the boundary elements.
        :return: float
        Solution of Poisson equation at a point (ksi, eta, zeta).
        """

        N = self.N
        nx = self.n_x
        ny = self.n_y
        nz = self.n_z

        sol = 0
        for i in range(N):
            d1, d2 = self.calculate_d(ksi, eta, zeta, i, nx[i], ny[i], nz[i])
            sol = sol + phi[i] * d2 - dphi[i] * d1

        sol = sol + c * (ksi ** 2 + eta ** 2 + zeta ** 2) / 6

        return sol

