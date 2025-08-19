import numpy as np
import matplotlib.pyplot as plt

# Input-output pairs
X = np.array([0.5, 1.5, 3.0, 4.0, 5.0])
D = np.array([8.0, 6.0, 5.0, 2.0, 0.5])

# Add bias column
X_prime = np.vstack([X, np.ones(len(X))]).T

# Solve for [w, b] using LLS
theta = np.linalg.inv(X_prime.T @ X_prime) @ X_prime.T @ D
w, b = theta

# Plot the fitting result
plt.scatter(X, D, label="Data points", color="blue")
plt.plot(X, w * X + b, label=f"LLS Fit: y = {w:.2f}x + {b:.2f}", color="red")
plt.xlabel("x")
plt.ylabel("d")
plt.legend()
plt.title("Linear Least-Squares Fit")
plt.grid(alpha=0.3)
plt.show()

print(f"LLS Solution: w = {w:.4f}, b = {b:.4f}")