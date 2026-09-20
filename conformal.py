import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def true_f(x):
    return np.sin(x)

def noise_std(x):
    return 0.1 + 0.3 * (x + 3) / 6   # heteroscedastic noise

def make_data(n):
    x = rng.uniform(-3, 3, n)
    y = true_f(x) + rng.normal(0, noise_std(x))
    return x, y

def fit_model(x, y):
    # simple polynomial fit
    coeffs = np.polyfit(x, y, 5)
    return lambda x_new: np.polyval(coeffs, x_new)

def split_conformal(x_train, y_train, x_cal, y_cal, x_test, y_test, alpha):
    model = fit_model(x_train, y_train)
    cal_resid = np.abs(y_cal - model(x_cal))
    n = len(cal_resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    threshold = np.sort(cal_resid)[k - 1]
    y_hat = model(x_test)
    lower = y_hat - threshold
    upper = y_hat + threshold
    covered = (y_test >= lower) & (y_test <= upper)
    return covered.mean()

# data
N = 2000
x, y = make_data(N)
idx = rng.permutation(N)
n_train = 200
n_cal = 500
x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
x_cal, y_cal = x[idx[n_train:n_train+n_cal]], y[idx[n_train:n_train+n_cal]]
x_test, y_test = x[idx[n_train+n_cal:]], y[idx[n_train+n_cal:]]

# sweep alpha
alphas = np.linspace(0.01, 0.20, 20)
achieved = [split_conformal(x_train, y_train, x_cal, y_cal, x_test, y_test, a) for a in alphas]
nominal = 1 - alphas

# plot
plt.figure(figsize=(6, 6))
plt.plot(nominal, achieved, "o-", label="split conformal")
plt.plot([0.8, 1.0], [0.8, 1.0], "k--", label="diagonal")
plt.xlabel("Nominal coverage (1 - alpha)")
plt.ylabel("Achieved coverage")
plt.title("Split Conformal Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("conformal_coverage.png", dpi=120)
plt.show()

# experiment: use training residuals instead of calibration
print("\n--- Training-residual experiment (should undercover) ---")
model = fit_model(x_train, y_train)
train_resid = np.abs(y_train - model(x_train))
for a in [0.05, 0.10, 0.15]:
    n = len(train_resid)
    k = int(np.ceil((n + 1) * (1 - a)))
    k = min(k, n)
    thr = np.sort(train_resid)[k - 1]
    y_hat = model(x_test)
    cov = ((y_test >= y_hat - thr) & (y_test <= y_hat + thr)).mean()
    print(f"alpha={a:.2f}  nominal={1-a:.2f}  achieved={cov:.3f}")