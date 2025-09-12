# q2_sequential_expanded.py, with the help of chatgpt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


SEED = 30
np.random.seed(SEED)
torch.manual_seed(SEED)


def target_function(x):
    return 1.2 * np.sin(np.pi * x) - np.cos(2.4 * np.pi * x)

# training domain: [-2, 2], step 0.05
x_train_np = np.arange(-2.0, 2.0 + 1e-12, 0.05)
y_train_np = target_function(x_train_np)

# test domain: [-2, 2], step 0.01 (dense)
x_test_np = np.arange(-2.0, 2.0 + 1e-12, 0.01)
y_test_np = target_function(x_test_np)

# Convert to torch tensors (column vectors)
X_train = torch.from_numpy(x_train_np.reshape(-1, 1).astype(np.float32))
Y_train = torch.from_numpy(y_train_np.reshape(-1, 1).astype(np.float32))
X_test = torch.from_numpy(x_test_np.reshape(-1, 1).astype(np.float32))
Y_test = torch.from_numpy(y_test_np.reshape(-1, 1).astype(np.float32))

# -----------------------
# simple 1 - N - 1 MLP with tanh
class SimpleNet(nn.Module):
    def __init__(self, N=10):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(1, N)
        self.activation = nn.Tanh()
        self.output = nn.Linear(N, 1)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

# training (sequential / online mode: batch_size = 1)
def train_sequential(model, X_train, Y_train, lr=0.01, epochs=1000, verbose=False):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    n = X_train.shape[0]
    loss_hist = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in perm:
            xi = X_train[i].unsqueeze(0)   # shape (1,1)
            yi = Y_train[i].unsqueeze(0)

            yhat = model(xi)
            loss = criterion(yhat, yi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_hist.append(epoch_loss / n)
        if verbose and (epoch % (max(1, epochs // 5)) == 0):
            print(f"Epoch {epoch+1}/{epochs}, avg loss={loss_hist[-1]:.6e}")
    return loss_hist

# evaluate helper
def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        yhat = model(X)
        total_error = torch.sum(torch.abs(yhat - Y)).item()
    return total_error, yhat.cpu().numpy().flatten()


hidden_sizes = [1, 2, 5, 10, 20, 50, 100]

# hyperparameters 
LR = 0.01
EPOCHS = 1000
VERBOSE = False


# Train all models first
results = {}
for N in hidden_sizes:
    torch.manual_seed(SEED)  
    model = SimpleNet(N)

    print(f"\nTraining 1-{N}-1 network (epochs={EPOCHS}, lr={LR}) ...")
    loss_hist = train_sequential(model, X_train, Y_train, lr=LR, epochs=EPOCHS, verbose=VERBOSE)
    train_error, yhat_train = evaluate(model, X_train, Y_train)
    test_error, yhat_test = evaluate(model, X_test, Y_test)

    # predictions at x = -3 and +3
    x_minus3 = torch.tensor([[-3.0]], dtype=torch.float32)
    x_plus3  = torch.tensor([[ 3.0]], dtype=torch.float32)
    with torch.no_grad():
        p_minus3 = model(x_minus3).item()
        p_plus3  = model(x_plus3).item()

    results[N] = {
        'model': model,
        'loss_hist': loss_hist,
        'train_error': train_error,
        'test_error': test_error,
        'yhat_test': yhat_test,
        'p_minus3': p_minus3,
        'p_plus3' : p_plus3
    }

    print(f"N={N:3d}  train_error={train_error:.6e}  test_error={test_error:.6e}  "
          f"pred(-3)={p_minus3:.6f}  pred(+3)={p_plus3:.6f}")

# Plot all graphs after training is complete
for N in hidden_sizes:
    r = results[N]
    plt.figure(figsize=(8,4))
    plt.plot(x_test_np, y_test_np, 'k-', linewidth=2, label='True function')
    plt.plot(x_test_np, r['yhat_test'], '-', label=f'Predicted (N={N})')
    plt.scatter(x_train_np, y_train_np, s=12, color='gray', alpha=0.4, label='train samples')
    plt.title(f'Function approximation (sequential) — 1-{N}-1')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# final summary printout
print("\nFinal summary:")
print(f"{'N':>3}  {'train_error':>12}  {'test_error':>12}  {'pred(-3)':>12}  {'pred(+3)':>12}  {'err(-3)':>12}  {'err(+3)':>12}  {'total_err':>12}")
for N in hidden_sizes:
    r = results[N]
    true_minus3 = target_function(-3.0)
    true_plus3 = target_function(3.0)
    err_minus3 = abs(r['p_minus3'] - true_minus3)
    err_plus3 = abs(r['p_plus3'] - true_plus3)
    total_err = err_minus3 + err_plus3
    print(f"{N:3d}  {r['train_error']:12.4e}  {r['test_error']:12.4e}  "
          f"{r['p_minus3']:12.6f}  {r['p_plus3']:12.6f}  "
          f"{err_minus3:12.6f}  {err_plus3:12.6f}  {total_err:12.6f}")