# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Perceptron Learning Algorithm
# -----------------------------
def perceptron_train(X, d, eta=1.0, epochs=20):
    """
    Train a perceptron using the perceptron learning rule.
    X: input matrix (N x d)
    d: desired outputs (N,)
    eta: learning rate
    epochs: number of passes over dataset
    Returns: weight trajectory (list of [w1, w2, b])
    """
    n_samples, n_features = X.shape
    w = np.random.randn(n_features) * 0.1  # small random init
    b = 0.0
    trajectory = [(w.copy(), b)]

    for _ in range(epochs):
        for i in range(n_samples):
            v = np.dot(w, X[i]) + b
            y = 1 if v > 0 else 0
            e = d[i] - y  #compute error
            # update
            w = w + eta * e * X[i]
            b = b + eta * e
            trajectory.append((w.copy(), b))

    return trajectory


# -----------------------------
# Logic Gate Datasets
# -----------------------------
def get_dataset(gate):
    if gate == "AND":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        d = np.array([0,0,0,1])
    elif gate == "OR":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        d = np.array([0,1,1,1])
    elif gate == "NAND":
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        d = np.array([1,1,1,0])
    elif gate == "NOT":
        X = np.array([[0],[1]])
        d = np.array([1,0])
    else:
        raise ValueError("Unknown gate")
    return X, d


# -----------------------------
# Plotting Function
# -----------------------------
def plot_trajectory(trajectory, title, epochs, samples_per_epoch):
    ws_all = np.array([w for w, b in trajectory])
    bs_all = np.array([b for w, b in trajectory])

    # Sample trajectory at epoch boundaries (including epoch 0)
    epoch_indices = [k * samples_per_epoch for k in range(epochs + 1)]
    ws = ws_all[epoch_indices]
    bs = bs_all[epoch_indices]

    plt.figure(figsize=(8,4))
    for i in range(ws.shape[1]):
        plt.plot(range(epochs + 1), ws[:, i], label=f"x{i+1}")
    plt.plot(range(epochs + 1), bs, label="bias b", linestyle="--")

    # Annotate final values at the last epoch without extending x-axis
    ax = plt.gca()
    lines = ax.get_lines()
    # Weights first
    for i in range(ws.shape[1]):
        final_val = ws[-1, i]
        color = lines[i].get_color() if i < len(lines) else None
        plt.annotate(f"{final_val:.3f}",
                     xy=(epochs, final_val),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, color=color)
    # Bias last
    final_bias = bs[-1]
    bias_idx = ws.shape[1]
    bias_color = lines[bias_idx].get_color() if bias_idx < len(lines) else None
    plt.annotate(f"{final_bias:.3f}",
                 xy=(epochs, final_bias),
                 xytext=(0, 8), textcoords="offset points",
                 ha="center", va="bottom", fontsize=9, color=bias_color)

    ax.set_xlim(0, epochs)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Weight Trajectory - {title}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# -----------------------------
# Run Experiments
# -----------------------------
for gate in ["AND", "OR", "NAND", "NOT"]:
    X, d = get_dataset(gate)
    epochs = 20
    traj = perceptron_train(X, d, eta=1.0, epochs=epochs)
    plot_trajectory(traj, gate, epochs=epochs, samples_per_epoch=X.shape[0])