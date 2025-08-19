import numpy as np
import matplotlib.pyplot as plt

# Input-output pairs
X = np.array([0.5, 1.5, 3.0, 4.0, 5.0])
D = np.array([8.0, 6.0, 5.0, 2.0, 0.5])

# LMS Algorithm
def lms_train(X, D, eta=0.02, epochs=100):
    w = np.random.randn()  # Random initialization
    b = np.random.randn()
    trajectory = [(w, b)]

    for _ in range(epochs):
        for i in range(len(X)):
            y = w * X[i] + b
            e = D[i] - y
            w += eta * e * X[i]
            b += eta * e
            trajectory.append((w, b))

    return trajectory

# Train LMS
eta = 0.02
epochs = 100
trajectory = lms_train(X, D, eta=eta, epochs=epochs)

# Extract weights and biases
weights, biases = zip(*trajectory)

# Plot LMS fitting result
plt.scatter(X, D, label="Data points", color="blue")
plt.plot(X, weights[-1] * X + biases[-1], label=f"LMS Fit: y = {weights[-1]:.2f}x + {biases[-1]:.2f}", color="green")
plt.xlabel("x")
plt.ylabel("d")
plt.legend()
plt.title("LMS Fit")
plt.grid(alpha=0.3)
plt.show()

# Plot weight and bias trajectories
plt.figure(figsize=(8, 4))
plt.plot(weights, label="w (weight)", color="red")
plt.plot(biases, label="b (bias)", color="blue", linestyle="--")
plt.xlabel("Update step")
plt.ylabel("Value")
plt.legend()
plt.title("Weight and Bias Trajectories (LMS)")
plt.grid(alpha=0.3)
plt.show()

print(f"LMS Final Solution: w = {weights[-1]:.4f}, b = {biases[-1]:.4f}")