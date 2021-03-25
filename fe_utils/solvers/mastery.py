"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import cos,sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser



def assemble(fs1,fs2, f):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""

    # Create an appropriate (complete) quadrature rule.

    # Tabulate the basis functions and their gradients at the quadrature points.

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!

    A = sp.lil_matrix((fs1.node_count, fs1.node_count))
    B = sp.lil_matrix((fs2.node_count, fs1.node_count))
    rhs = np.zeros(fs1.node_count + fs2.node_count)
    l = rhs[:fs1.node_count]

    # Now loop over all the cells and assemble A and l
    fe1 = fs1.element
    fe2 = fs2.element
    d = fs1.element.cell.dim
    mesh = fs1.mesh

    # Create a quadrature rule which is exact for (f1-f2)**2.
    Q = gauss_quadrature(fe1.cell, 2*max(fe1.degree, fe2.degree))

    # Evaluate the local basis functions at the quadrature points.
    phi1 = fe1.tabulate(Q.points)
    phi1_grad = fe1.tabulate(Q.points, grad=True)
    phi2 = fe2.tabulate(Q.points)
    phi2_grad = fe2.tabulate(Q.points, grad=True)

    val = 0.
    for c in range(mesh.entity_counts[-1]):
        # Find the appropriate global node numbers for this cell.
        nodes1 = fs1.cell_nodes[c, :]
        nodes2 = fs2.cell_nodes[c, :]

        # Compute the change of coordinates.
        J = mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))

        # i = basis, a = dimension, q= quadrature 
        f_quadrature = np.einsum("i, iaq -> aq", f.values[nodes1], phi1.T)
        l[nodes1] += np.einsum("iaq, aq -> i", phi1.T, f_quadrature * Q.weights) * detJ 

        
        div = np.einsum("ab,qbia->iq", np.linalg.inv(J), phi1_grad)
        weighted_grad = np.einsum("q, iq ->iq", Q.weights, div)
        B[np.ix_(nodes2, nodes1)] += np.einsum("qi, jq ->ij", phi2, weighted_grad) * detJ


        half_grad_term = np.einsum("bc,qaic -> qaib", np.linalg.inv(J.T), phi1_grad)
        half_grad_termT = np.einsum("qaib -> qbia", half_grad_term)
        grad_term = (1/2)*(half_grad_term + half_grad_termT)
        weighted_grad = np.einsum("q, qaib->qaib", Q.weights, grad_term)
        A[np.ix_(nodes1, nodes1)] +=  np.einsum("qaib, qajb ->ij", weighted_grad, grad_term)* detJ 
        
    return A,B,rhs


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    f = Function(fs)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.
    def on_boundary_vec(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return [1.,1.]
        else:
            return [0.,0.]

    if isinstance(fs.element, VectorFiniteElement):
        f.interpolate(on_boundary_vec)
    else:
        f.interpolate(on_boundary)

    return np.flatnonzero(f.values)


def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given resolution. It
    should return both the solution :class:`~fe_utils.function_spaces.Function` and
    the :math:`L^2` error in the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """
    
    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, 2)
    ve = VectorFiniteElement(fe)
    qe = LagrangeElement(mesh.cell, 1)
    V = FunctionSpace(mesh, ve)
    Q = FunctionSpace(mesh, qe)

    # # Create a function to hold the analytic solution for comparison purposes.
    u_analytic_answer = Function(V)
    u_analytic_answer.interpolate(lambda x: (2*pi*(1 - cos(2*pi*x[0]))*sin(2*pi*x[1]),
                                            -2*pi*(1 - cos(2*pi*x[1]))*sin(2*pi*x[0])))
    p_analytic_answer = Function(Q)
    p_analytic_answer.interpolate(lambda x: 0)
    

    # # If the analytic answer has been requested then bail out now.
    if analytic:
        return (u_analytic_answer, p_analytic_answer), 0.0

    
    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(V)
    first = lambda x,y: -4*pi**3*cos(2*pi*x)*sin(2*pi*y) - 4*pi**3*sin(2*pi*y)*(cos(2*pi*x) - 1)
    second = lambda x,y: 4*pi**3*cos(2*pi*y)*sin(2*pi*x) + 4*pi**3*sin(2*pi*x)*(cos(2*pi*y) - 1)
    f.interpolate(lambda x: (first(x[0],x[1]), second(x[0],x[1])))

    # Assemble the finite element system.
    A,B,l = assemble(V,Q,f)

    # Create the function to hold the solution.
    zero = np.zeros((B.shape[0],B.shape[0]))
    u = Function(V)
    p = Function(Q)
    M = sp.bmat([[A, B.T],[B, zero]], format='lil')

    boundary1 = boundary_nodes(V)
    boundary1 = np.append(boundary1, [V.node_count + Q.node_count - 1])
    l[boundary1] = 0
    M[boundary1] = np.zeros((len(boundary1), V.node_count + Q.node_count))
    M[boundary1, boundary1] = np.ones(len(boundary1))


    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    M = sp.csc_matrix(M)
    lu = sp.linalg.splu(M)
    res = lu.solve(l)
    u.values[:] = res[: V.node_count]
    p.values[:] = res[V.node_count :]

    # Compute the L^2 error in the solution for testing purposes.
    u_error = vectorerrornorm(u_analytic_answer, u)
    p_error = errornorm(p_analytic_answer, p)
    error = (u_error**2 + p_error**2)**0.5

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return (u,p), error


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve the mastery problem.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    analytic = args.analytic
    plot_error = args.error

    (u,p), error = solve_mastery(resolution, analytic, plot_error)

    u.plot()
    p.plot()
