# rosenbrock_gd_newton.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la

# Rosenbrock function, gradient and Hessian
def rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

def grad_rosenbrock(xy):
    x, y = xy
    dfdx = -2.0 * (1 - x) - 400.0 * x * (y - x ** 2)
    dfdy = 200.0 * (y - x ** 2)
    return np.array([dfdx, dfdy])

def hess_rosenbrock(xy):
    x, y = xy
    d2fdx2 = 2.0 - 400.0 * (y - x ** 2) + 800.0 * x ** 2
    d2fdy2 = 200.0
    d2fdxdy = -400.0 * x
    return np.array([[d2fdx2, d2fdxdy],
                     [d2fdxdy, d2fdy2]])

def gradient_descent(x0, eta, max_iters=20000, tol=1e-8):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    vals = [rosenbrock(x)]
    for k in range(max_iters):
        g = grad_rosenbrock(x)
        x = x - eta * g
        traj.append(x.copy())
        vals.append(rosenbrock(x))
        if vals[-1] < tol:
            return np.array(traj), np.array(vals), k + 1
        if not np.isfinite(vals[-1]):
            return np.array(traj), np.array(vals), None
    return np.array(traj), np.array(vals), None

def newton_method(x0, max_iters=2000, tol=1e-12, mu0=1e-6, max_damping=1e6):
    """
    Newton's method with Levenberg-like damping (regularization).
    Solves H Δw = -g. If H is singular/indefinite/invert fails, add mu*I and
    increase mu until step is acceptable or mu exceeds max_damping.
    """
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    vals = [rosenbrock(x)]
    mu = mu0
    for k in range(max_iters):
        g = grad_rosenbrock(x)
        H = hess_rosenbrock(x)
        # try to solve (H + mu I) p = -g
        success = False
        for _ in range(20):  # try up to 20 increases of mu
            try:
                H_reg = H + mu * np.eye(2)
                p = la.solve(H_reg, -g, assume_a='gen')
                success = True
                break
            except la.LinAlgError:
                mu *= 10.0
                if mu > max_damping:
                    break
        if not success:
            # can't solve, abort
            return np.array(traj), np.array(vals), None

        # line search / backtracking to ensure sufficient decrease
        alpha = 1.0
        c = 1e-4
        f_curr = vals[-1]
        # try at most 20 backtracking steps
        for _ in range(20):
            x_new = x + alpha * p
            f_new = rosenbrock(x_new)
            if f_new <= f_curr + c * alpha * np.dot(g, p):
                break
            alpha *= 0.5
        else:
            # backtracking failed -> increase damping and continue
            mu *= 10.0
            continue

        # accept step
        x = x + alpha * p
        traj.append(x.copy())
        vals.append(rosenbrock(x))

        # reduce mu if step was successful (encourage Newton-like steps)
        mu = max(mu0, mu * 0.1)

        if vals[-1] < tol:
            return np.array(traj), np.array(vals), k + 1
        if not np.isfinite(vals[-1]):
            return np.array(traj), np.array(vals), None

    return np.array(traj), np.array(vals), None

# plotting helpers (kept similar to your original)
def plot_results_comparison(traj_gd, vals_gd, iters_gd, traj_nt, vals_nt, iters_nt, eta, start):
    # grid for Rosenbrock
    xs = np.linspace(-1.5, 1.5, 400)
    ys = np.linspace(-0.5, 2.0, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X) ** 2 + 100.0 * (Y - X ** 2) ** 2

    fig = plt.figure(figsize=(18, 6))

    # GD contour + trajectory
    ax1 = fig.add_subplot(1, 3, 1)
    levels = np.logspace(-0.5, 3.5, 30)
    ax1.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')
    ax1.plot(traj_gd[:, 0], traj_gd[:, 1], '-o', color='red', markersize=3, linewidth=1, label='GD traj')
    ax1.plot(1, 1, 'g*', markersize=12, label='global minimum (1,1)')
    ax1.scatter([start[0]], [start[1]], color='orange', s=50, label='start')
    ax1.set_title(f'Gradient Descent (η={eta})\nconverged in {iters_gd if iters_gd is not None else "N/A"} iters')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_xlim([-1.5, 1.5]); ax1.set_ylim([-0.5, 2.0])
    ax1.set_aspect('equal', 'box')
    ax1.legend()

    # Newton contour + trajectory
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')
    ax2.plot(traj_nt[:, 0], traj_nt[:, 1], '-o', color='blue', markersize=3, linewidth=1, label='Newton traj')
    ax2.plot(1, 1, 'g*', markersize=12, label='global minimum (1,1)')
    ax2.scatter([start[0]], [start[1]], color='orange', s=50, label='start')
    ax2.set_title(f"Newton's Method\nconverged in {iters_nt if iters_nt is not None else 'N/A'} iters")
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_xlim([-1.5, 1.5]); ax2.set_ylim([-0.5, 2.0])
    ax2.set_aspect('equal', 'box')
    ax2.legend()

    # Newton's method: function value vs iteration (approach to minimum)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(vals_nt + 1e-20, '-ob', markersize=3, label="Newton f(x)")
    if iters_nt is not None:
        ax3.axvline(iters_nt, color='b', linestyle='--', label=f'converged @ {iters_nt}')
    ax3.set_title("Newton's Method: f(x) over iterations")
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('f(x,y) (log scale)')
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)         # for reproducibility
    start = np.random.rand(2)  # random start in (0,1) for x and y
    print("Random start (x,y):", start)

    # Part (a): Gradient descent
    eta_small = 0.001
    traj_small, vals_small, iters_small = gradient_descent(start, eta_small,
                                                          max_iters=200000,
                                                          tol=1e-12)
    if iters_small is not None:
        print(f"GD η={eta_small}: converged in {iters_small} iterations, "
              f"final f={vals_small[-1]:.3e}, final (x,y)={traj_small[-1]}")
    else:
        print(f"GD η={eta_small}: did not converge within max iterations. final f={vals_small[-1]:.3e}")

    # Part (b): Newton's method
    traj_newton, vals_newton, iters_newton = newton_method(start,
                                                          max_iters=2000,
                                                          tol=1e-12,
                                                          mu0=1e-6)
    if iters_newton is not None:
        print(f"Newton: converged in {iters_newton} iterations, "
              f"final f={vals_newton[-1]:.3e}, final (x,y)={traj_newton[-1]}")
    else:
        print(f"Newton: did not converge or aborted. final f={vals_newton[-1]:.3e}")

    # Comparison plot
    plot_results_comparison(traj_small, vals_small, iters_small,
                            traj_newton, vals_newton, iters_newton,
                            eta_small, start)