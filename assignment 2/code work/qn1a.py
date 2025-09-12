# rosenbrock_gd.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Rosenbrock function and gradient
def rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

def grad_rosenbrock(xy):
    x, y = xy
    dfdx = -2.0 * (1 - x) - 400.0 * x * (y - x ** 2)
    dfdy = 200.0 * (y - x ** 2)
    return np.array([dfdx, dfdy])

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
            return np.array(traj), np.array(vals), k + 1  # converged
        # early stopping for NaN/inf (divergence)
        if not np.isfinite(vals[-1]):
            return np.array(traj), np.array(vals), None
    return np.array(traj), np.array(vals), None  # reached max iters

def plot_results_3d(traj, vals, eta, iters_taken, start, 
                   x_range=(-1.5, 1.5), y_range=(-0.5, 2.0)):
    """
    Plot 3D visualization of Rosenbrock function with gradient descent trajectory
    
    Parameters:
    - traj: trajectory points
    - vals: function values
    - eta: learning rate
    - iters_taken: iterations to convergence
    - start: starting point
    - x_range: tuple (x_min, x_max) for x-axis boundaries
    - y_range: tuple (y_min, y_max) for y-axis boundaries
    """
    # 3D surface plot of Rosenbrock
    xs = np.linspace(x_range[0], x_range[1], 100)
    ys = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X) ** 2 + 100.0 * (Y - X ** 2) ** 2

    fig = plt.figure(figsize=(15, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                           norm=LogNorm(), linewidth=0, antialiased=True)
    
    # Plot trajectory in 3D - filter out non-finite values to avoid projection warnings
    traj_z = np.array([rosenbrock(point) for point in traj])
    finite_mask = np.isfinite(traj_z) & np.isfinite(traj[:, 0]) & np.isfinite(traj[:, 1])
    traj_f = traj[finite_mask]
    traj_z_f = traj_z[finite_mask]
    if traj_f.size > 0:
        ax1.plot(traj_f[:, 0], traj_f[:, 1], traj_z_f, '-o', color='red',
                 markersize=4, linewidth=3, label='trajectory', alpha=0.9)
    
    # Mark start and end points with larger markers
    ax1.scatter([start[0]], [start[1]], [rosenbrock(start)], 
                color='orange', s=150, label='start', edgecolors='black', linewidth=1)
    ax1.scatter([1], [1], [0], color='green', s=150, label='global min', 
                edgecolors='black', linewidth=1)
    
    # Add trajectory points as individual markers for better visibility
    if traj_f.size > 0:
        ax1.scatter(traj_f[:, 0], traj_f[:, 1], traj_z_f, color='red', s=20, alpha=0.7)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title(f'3D Rosenbrock Surface (η={eta})')
    ax1.legend()
    
    # Adjust viewpoint to be slightly more from the top
    ax1.view_init(elev=45, azim=60)  # elev=elevation angle, azim=azimuth angle

    # 2D contour plot (for comparison)
    ax2 = fig.add_subplot(1, 3, 2)
    levels = np.logspace(-0.5, 3.5, 25)
    ax2.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')
    ax2.plot(traj[:, 0], traj[:, 1], '-o', color='red', markersize=3, linewidth=1)
    ax2.plot(1, 1, 'g*', markersize=12, label='global minimum (1,1)')
    ax2.scatter([start[0]], [start[1]], color='orange', s=50, label='start')
    ax2.set_title(f'2D Contour View (η={eta})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-0.5, 2.0])
    ax2.set_aspect('equal', 'box')

    # Function value vs iteration
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(vals + 1e-20, label=f'f(x) (η={eta})')  # add label for legend
    ax3.set_title(f'Function value vs iterations (η={eta})')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('f(x,y) (log scale)')
    if iters_taken is not None:
        ax3.axvline(iters_taken, color='gray', linestyle='--',
                    label=f'converged in {iters_taken} iters')
    ax3.legend()

    plt.tight_layout()
    plt.show()

def plot_results(traj, vals, eta, iters_taken, start):
    # contour plot of Rosenbrock
    xs = np.linspace(-1.5, 1.5, 400)
    ys = np.linspace(-0.5, 2.0, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X) ** 2 + 100.0 * (Y - X ** 2) ** 2

    plt.figure(figsize=(12, 5))

    # Trajectory on contours
    plt.subplot(1, 2, 1)
    levels = np.logspace(-0.5, 3.5, 25)
    plt.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='viridis')
    plt.plot(traj[:, 0], traj[:, 1], '-o', color='red', markersize=3, linewidth=1)
    plt.plot(1, 1, 'g*', markersize=12, label='global minimum (1,1)')
    plt.scatter([start[0]], [start[1]], color='orange', s=50, label='start')
    plt.title(f'Trajectory on Rosenbrock contours (η={eta})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis([-1.5, 1.5, -0.5, 2.0])
    plt.gca().set_aspect('equal', 'box')

    # Function value vs iteration
    plt.subplot(1, 2, 2)
    plt.semilogy(vals + 1e-20)  # log scale; small eps to avoid log(0)
    plt.title(f'Function value vs iterations (η={eta})')
    plt.xlabel('Iteration')
    plt.ylabel('f(x,y) (log scale)')
    if iters_taken is not None:
        plt.axvline(iters_taken, color='gray', linestyle='--',
                    label=f'converged in {iters_taken} iters')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)         # for reproducibility
    start = np.random.rand(2)  # random start in (0,1) for x and y
    print("Random start (x,y):", start)

    # Experiment 1: small learning rate (expected to converge slowly but stably)
    eta_small = 0.001
    traj_small, vals_small, iters_small = gradient_descent(start, eta_small,
                                                          max_iters=200000,
                                                          tol=1e-12)
    if iters_small is not None:
        print(f"η={eta_small}: converged in {iters_small} iterations, "
              f"final f={vals_small[-1]:.3e}, final (x,y)={traj_small[-1]}")
    else:
        print(f"η={eta_small}: did not converge within max iterations. "
              f"final f={vals_small[-1]:.3e}")

    # Use 3D plotting for better visualization
    print(f"\n=== Small Learning Rate (η={eta_small}) ===")
    plot_results_3d(traj_small, vals_small, eta_small, iters_small, start)
    
    # # Optional: Zoomed view around the minimum
    # print("Zoomed view around minimum:")
    # plot_results_3d(traj_small, vals_small, eta_small, iters_small, start,
    #                x_range=(0.5, 1.5), y_range=(0.5, 1.5))

    # Experiment 2: large learning rate (likely to diverge or oscillate)
    eta_large = 0.5
    traj_large, vals_large, iters_large = gradient_descent(start, eta_large,
                                                          max_iters=20000,
                                                          tol=1e-12)
    if iters_large is not None:
        print(f"η={eta_large}: converged in {iters_large} iterations, "
              f"final f={vals_large[-1]:.3e}, final (x,y)={traj_large[-1]}")
    else:
        final_status = ("diverged" if not np.isfinite(vals_large[-1])
                        else "did not converge within max iterations")
        print(f"η={eta_large}: {final_status}. final f={vals_large[-1]:.3e}")

    print(f"\n=== Large Learning Rate (η={eta_large}) ===")
    plot_results_3d(traj_large, vals_large, eta_large, iters_large, start)