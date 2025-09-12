import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---------- USER: put your target function here ----------
# Example default: f(x) = sin(pi * x)
def target_function(x):
    return 1.2 * np.sin(np.pi * x) - np.cos(2.4 * np.pi * x)
# --------------------------------------------------------

def make_data(xmin=-2.0, xmax=2.0, step=0.05):
    x = np.arange(xmin, xmax + 1e-12, step)
    y = target_function(x)
    return x.reshape(-1, 1), y.reshape(-1, 1)

def unpack_params(params, n_hidden):
    """
    params layout:
      w1: (n_hidden,)   input->hidden weights
      b1: (n_hidden,)   hidden biases
      w2: (n_hidden,)   hidden->output weights
      b2: (1,)          output bias
    total length = 3*n_hidden + 1
    """
    n = n_hidden
    w1 = params[0:n].reshape((1, n))        # shape (1, n)
    b1 = params[n:2*n].reshape((1, n))      # shape (1, n)
    w2 = params[2*n:3*n].reshape((n, 1))    # shape (n, 1)
    b2 = params[-1].reshape((1,)) if isinstance(params[-1], np.ndarray) else np.array([params[-1]])
    return w1, b1, w2, b2

def mlp_forward(x, params, n_hidden):
    # x: (N,1)
    w1, b1, w2, b2 = unpack_params(params, n_hidden)
    # hidden pre-activation: (N, n)
    z = np.tanh(np.dot(x, w1) + b1)  # tanh activation
    # output: (N,1)
    y = np.dot(z, w2) + b2
    return y

def residuals(params, x, y_true, n_hidden):
    y_pred = mlp_forward(x, params, n_hidden)
    # return flattened residuals for least_squares
    return (y_pred - y_true).ravel()

def init_params(n_hidden, scale=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    # 3*n + 1 parameters
    return rng.normal(scale=scale, size=(3 * n_hidden + 1,))

def train_mlp_least_squares(x_train, y_train, n_hidden, max_nfev=1000):
    p0 = init_params(n_hidden, scale=0.5, random_state=42)
    # method='lm' (Levenberg-Marquardt)
    # Ensure number of residuals >= number of variables (here: len(train) >= 3n+1)
    res = least_squares(residuals, p0, args=(x_train, y_train, n_hidden),
                        method='lm', max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
    return res

def mse(a, b):
    return np.mean((a - b) ** 2)

def run_experiments(hidden_list=None):
    if hidden_list is None:
        hidden_list = [1,2,5,10,20]

    x_train, y_train = make_data(step=0.05)
    x_test, y_test = make_data(step=0.01)

    results = []

    for n in hidden_list:
        params_count = 3 * n + 1
        if params_count > x_train.shape[0]:
            print(f"Skipping n={n}: params ({params_count}) > train samples ({x_train.shape[0]})")
            continue

        print(f"\nTraining 1-{n}-1 network (params={params_count}) on {x_train.shape[0]} samples")
        res = train_mlp_least_squares(x_train, y_train, n_hidden=n, max_nfev=2000)

        y_train_pred = mlp_forward(x_train, res.x, n)
        y_test_pred = mlp_forward(x_test, res.x, n)

        train_mse = mse(y_train_pred, y_train)
        test_mse = mse(y_test_pred, y_test)
        
        # Calculate train and test errors (L1 sum like qn2a.py)
        train_error = np.sum(np.abs(y_train_pred - y_train))
        test_error = np.sum(np.abs(y_test_pred - y_test))
        
        # Predictions at x = -3 and +3
        x_minus3 = np.array([[-3.0]])
        x_plus3 = np.array([[3.0]])
        p_minus3 = mlp_forward(x_minus3, res.x, n)[0, 0]
        p_plus3 = mlp_forward(x_plus3, res.x, n)[0, 0]

        print(f"  success={res.success}, status={res.status}, message='{res.message}'")
        print(f"  nfev={res.nfev}, train_MSE={train_mse:.6e}, test_MSE={test_mse:.6e}")
        print(f"N={n:3d}  train_error={train_error:.6e}  test_error={test_error:.6e}  "
              f"pred(-3)={p_minus3:.6f}  pred(+3)={p_plus3:.6f}")

        results.append({
            'n': n,
            'res': res,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_error': train_error,
            'test_error': test_error,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'x_train': x_train,
            'x_test': x_test
        })

        # Plotting: target vs model on test set
        plt.figure(figsize=(6, 4))
        plt.plot(x_test.ravel(), y_test.ravel(), label='target', color='C0')
        plt.plot(x_test.ravel(), y_test_pred.ravel(), label=f'1-{n}-1 MLP', linestyle='--', color='C1')
        plt.title(f'Function approximation (batch LM) — 1-{n}-1')
        plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results

if __name__ == '__main__':
    # Run experiments (this will show plots for each hidden size)
    results = run_experiments()
    
    # Final summary printout (same format as qn2a.py)
    print("\nFinal summary:")
    print(f"{'N':>3}  {'train_error':>12}  {'test_error':>12}  {'pred(-3)':>12}  {'pred(+3)':>12}  {'err(-3)':>12}  {'err(+3)':>12}  {'total_err':>12}")
    
    for r in results:
        n = r['n']
        # Calculate errors at x = -3 and +3
        true_minus3 = target_function(-3.0)
        true_plus3 = target_function(3.0)
        x_minus3 = np.array([[-3.0]])
        x_plus3 = np.array([[3.0]])
        p_minus3 = mlp_forward(x_minus3, r['res'].x, n)[0, 0]
        p_plus3 = mlp_forward(x_plus3, r['res'].x, n)[0, 0]
        err_minus3 = abs(p_minus3 - true_minus3)
        err_plus3 = abs(p_plus3 - true_plus3)
        total_err = err_minus3 + err_plus3
        
        print(f"{n:3d}  {r['train_error']:12.4e}  {r['test_error']:12.4e}  "
              f"{p_minus3:12.6f}  {p_plus3:12.6f}  "
              f"{err_minus3:12.6f}  {err_plus3:12.6f}  "
              f"{total_err:12.6f}")
    
    # Example: summarize best test MSE
    best = min(results, key=lambda r: r['test_mse'])
    print(f"\nBest test MSE: n={best['n']} -> {best['test_mse']:.6e}")