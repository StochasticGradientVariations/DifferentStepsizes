import numpy as np
import scipy.linalg as LA
import scipy.sparse as spr
import scipy.sparse.linalg as spr_LA
from time import perf_counter


def safe_division(x, y):
    """
    Computes safe division x/y for small positive values x and y
    """
    return np.exp(np.log(x) - np.log(y)) if y != 0 else 1e16


# def ad_grad(J, df, x0, la_0=1e-6, numb_iter=100):
#     """
#     Minimize f(x) by adaptive gradient method.
#     Takes J as some evaluation function for comparison.
#
#     """
#     begin = perf_counter()
#     x_old = x0
#     grad_old = df(x0)
#     x = x0 - la_0 * grad_old
#     la_old = 1
#     th = 1e9
#     steps_array = []
#     values = [J(grad_old)]
#
#     for i in range(numb_iter):
#         grad = df(x)
#         norm_x = LA.norm(x - x_old)
#         norm_grad = LA.norm(grad - grad_old)
#         #la = min(np.sqrt(1 + th) * la_old,  0.5 * norm_x / norm_grad)
#         la = min(np.sqrt(1 + th) * la_old,  0.5 * safe_division(norm_x, norm_grad))
#         th = la / la_old
#         x_old = x.copy()
#         x -= la * grad
#         la_old = la
#         grad_old = grad
#         values.append(J(grad))
#         steps_array.append(la)
#     end = perf_counter()
#
#     print("Time execution of adaptive gradient descent:", end - begin)
#     return values, x, steps_array

def ad_grad(J, df, x0, la_0=1e-6, numb_iter=100):
    """
    Minimize f(x) by adaptive gradient method, με τ1 = (k+1)/k * la_old
    και τ2 = 0.5 * (norm_x / norm_grad).
    """
    begin     = perf_counter()
    x_old     = x0.copy()
    grad_old  = df(x0)
    x         = x0 - la_0 * grad_old
    la_old    = la_0
    steps     = []
    values    = [J(grad_old)]

    for i in range(numb_iter):
        k        = i + 1
        grad     = df(x)
        norm_x   = LA.norm(x - x_old)
        norm_g   = LA.norm(grad - grad_old)

        # νέα τ1 και τ2
        tau1     = (k+1)/k * la_old
        tau2     = 0.5 * safe_division(norm_x, norm_g)
        la       = min(tau1, tau2)

        # ενημερωσεις
        x_old    = x.copy()
        x       -= la * grad
        la_old   = la
        grad_old = grad

        values.append(J(grad))
        steps.append(la)

    end = perf_counter()
    print("Time execution of adaptive gradient descent:", end - begin)
    return values, x, steps

# def ad_grad_mod(J, df, x0, la_0=1e-6, numb_iter=100):
#     """
#     Minimize f(x) by adaptive gradient method with custom tau1 update (k/(k+3)).
#     J   : evaluation function (e.g., f(x) or loss)
#     df  : gradient function df(x) = ∇f(x)
#     x0  : initial point
#     la_0: initial small step
#     numb_iter: number of iterations
#     Returns: values history, final x, steps history
#     """
#     begin = perf_counter()
#
#     # Initialization
#     x_old     = x0.copy()
#     grad_old  = df(x0)
#     x         = x0 - la_0 * grad_old
#     la_old    = la_0
#
#     values      = [J(grad_old)]
#     steps_array = []
#
#     # Main loop
#     for k in range(1, numb_iter + 1):
#         grad      = df(x)
#         norm_x    = LA.norm(x - x_old)
#         norm_grad = LA.norm(grad - grad_old)
#
#         # Custom τ1: k/(k+3) * la_old
#         tau1 = (k) / (k + 3) * la_old
#         # τ2: local curvature bound with factor 0.5
#         tau2 = 0.5 * safe_division(norm_x, norm_grad)
#         la   = min(tau1, tau2)
#
#         # Update variables
#         x_old     = x.copy()
#         x        -= la * grad
#         la_old    = la
#         grad_old  = grad
#
#         # Record history
#         values.append(J(grad))
#         steps_array.append(la)
#
#     end = perf_counter()
#     print("Time execution with custom tau1 (k/(k+3)) update:", end - begin)
#
#     return values, x, steps_array


def ad_grad_tighter(J, df, x0, la_0=1e-6, numb_iter=100):
    """
    Minimize f(x)  by adaptive gradient method with tighter estimate.
    Takes J as some evaluation function for comparison.

    """
    begin = perf_counter()
    values, x_old, steps_array = ad_grad(J, df, x0, la_0, 10)
    th = 1
    la_old = steps_array[-1]
    grad_old = df(x_old)
    x = x_old - la_old * grad_old

    for i in range(numb_iter):
        grad = df(x)
        norm_x = LA.norm(x - x_old)
        norm_grad = LA.norm(grad - grad_old)
        denom = 3 * LA.norm(grad)**2 - 4 * np.vdot(grad, grad_old)
        if denom > 0:
            la = min(np.sqrt(1 + th) * la_old,   safe_division(norm_x, np.sqrt(denom)))
        else:
            la = np.sqrt(1+th) * la_old
        th = la / la_old
        x_old = x.copy()
        x -= la * grad
        la_old = la
        grad_old = grad
        values.append(J(grad))
        steps_array.append(la)
    end = perf_counter()

    print("Time execution of adaptive gradient descent:", end - begin)
    return np.array(values), x, steps_array



def ad_grad_smooth(J, df, x0, L, numb_iter=100):
    """
    Minimize f(x)  by adaptive gradient method knowing L.
    Takes J as some evaluation function for comparison.

    """
    begin = perf_counter()
    x_old = x0
    grad_old = df(x0)
    la_old = 1./L
    th = 1e9
    x = x0 - la_old * grad_old
    values = [J(grad_old)]
    steps_array = []

    for i in range(numb_iter):
        grad = df(x)
        norm_x = LA.norm(x - x_old)
        norm_grad = LA.norm(grad - grad_old)
        la = min(
            np.sqrt(1 + th) * la_old,  1 / (la_old * L**2) + 0.5 * safe_division(norm_x, norm_grad))
        th = la / la_old
        x_old = x.copy()
        x -= la * grad
        la_old = la
        grad_old = grad
        values.append(J(grad))
        steps_array.append(la)
    end = perf_counter()

    print("Time execution of adaptive gradient descent (L is known):", end - begin)
    return np.array(values), x, steps_array

def ad_grad_accel(J, df, x0, la_0=1e-6, numb_iter=100):
    """
    Minimize f(x)  by heuristic accelerated adaptive gradient method.
    Takes J as some evaluation function for comparison.

    """
    begin = perf_counter()
    x_old = x0.copy()
    y_old = x_old
    grad_old = df(x0)
    x = x0 - la_0 * grad_old
    y = x
    la_old = 1
    Lambda_old = 1
    th = 1e9
    Th = 1e9
    values = [J(grad_old)]
    steps_array = []
    for i in range(numb_iter):
        grad = df(y)
        norm_x = LA.norm(y - y_old)
        norm_grad = LA.norm(grad - grad_old)
        la = min(np.sqrt(1 + th) * la_old,  0.5 * safe_division(norm_x, norm_grad))
        Lambda = min(
            np.sqrt(1 + Th) * Lambda_old,  0.5 / safe_division(norm_x, norm_grad))
        th = la / la_old
        Th = Lambda / Lambda_old
        t = np.sqrt(Lambda * la)
        beta = (1 - t) / (1 + t)
        x = y - la * grad
        y_old = y
        y = x + beta * (x - x_old)
        x_old = x.copy()
        la_old = la
        Lambda_old = Lambda
        grad_old = grad
        values.append(J(grad))
        steps_array.append(la)
    end = perf_counter()

    print("Time execution of accelerated adaptive gradient descent:", end - begin)
    return np.array(values), x, steps_array


# GD #
def gd(J, df, x0, la=1, numb_iter=100):
    """
    Gradient descent for minimizing smooth f.
    """
    begin = perf_counter()
    x = x0.copy()
    values = [J(df(x0))]

    for i in range(numb_iter):
        grad = df(x)
        x -=  la * grad
        values.append(J(grad))
    end = perf_counter()
    print("Time execution for GD:", end - begin)
    return np.array(values), x


# accelerated GD
def accel_gd(J, df, x0, la, numb_iter=100):
    """
    Accelerated (Nesterov) gradient descent for minimizing smooth f.
    """
    begin = perf_counter()
    x, y = x0.copy(), x0.copy()
    t = 1.
    values = [J(df(x0))]

    for i in range(numb_iter):
        grad = df(y)
        x1 = y - la * grad
        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x1 + (t - 1) / t1 * (x1 - x)
        values.append(J(grad))
        x, t = x1, t1
    end = perf_counter()
    print("Time execution for accelerated GD:", end - begin)
    return np.array(values), x


def accel_str_gd(J, df, x0, la, mu, numb_iter=100):
    """
    Accelerated (Nesterov) gradient descent for minimizing smooth strongly convex f.
    """
    begin = perf_counter()
    x, y = x0.copy(), x0.copy()
    kappa = np.sqrt((1/la) / mu)
    beta = (kappa - 1) / (kappa + 1)
    values = [J(df(x0))]

    for i in range(numb_iter):
        grad = df(x)
        y1 = x - la * grad
        x = y1 + beta * (y1 - y)
        values.append(J(grad))
        y = y1
    end = perf_counter()
    print("Time execution for accelerated GD:", end - begin)
    return np.array(values), x
