"""
Functions
---------
.. autosummary::
   :toctree: generated/
    line_search_wolfe2
    scalar_search_wolfe2

"""
from warnings import warn

import numpy as np


class LineSearchWarning(RuntimeWarning):
    pass


def _check_c1_c2(c1, c2):
    if not (0 < c1 < c2 < 1):
        raise ValueError("'c1' and 'c2' do not satisfy"
                         "'0 < c1 < c2 < 1'.")


#------------------------------------------------------------------------------
# Armijo line and scalar searches
#------------------------------------------------------------------------------


def line_search_BFGS(f, myfprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, amax=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction. The search direction must be a descent direction
        for the algorithm to converge.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.

    """

    def phi(alpha):
        return f(xk + alpha * pk, *args)

    old_fval = phi(0)
    fprime = myfprime

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)

    alpha_star, phi_star = scalar_search_armijo(
            phi, old_fval, derphi0, c1 = c1, alpha0 = amax, amin = 0, maxiter = maxiter)

    if alpha_star is None:
        warn('The line search algorithm did not converge',
             LineSearchWarning, stacklevel=2)

    return alpha_star, phi_star

def scalar_search_armijo(phi, phi0, derphi0, c1 = 1e-4, alpha0 = 1, amin = 0, maxiter = 10):
    """Minimize over alpha, the function ``phi(alpha)``.

    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57

    alpha > 0 is assumed to be a descent direction.

    Returns
    -------
    alpha
    phi1

    """
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    iter = 0
    while alpha1 > amin and iter < maxiter:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
        iter += 1

    # Failed to find a suitable step length
    return None, phi_a1

line_search = line_search_BFGS
