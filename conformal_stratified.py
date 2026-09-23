import numpy as np

def true_f(x):
    return np.sin(x)

def noise_std(x):
    return 0.1 + 0.3 * (x + 3) / 6

def make_data(n, rng):
    x = rng.uniform(-3, 3, n)
    y = true_f(x) + rng.normal(0, noise_std(x))
    return x, y

def fit_model(x, y, degree=15):
    coeffs = np.polyfit(x, y, degree)
    return lambda x_new: np.polyval(coeffs, x_new)

def one_run(rng, alpha, n_train=30, n_cal=500, n_test=1300):
    # generate ONE pool of data
    N = n_train + n_cal + n_test
    x, y = make_data(N, rng)
    idx = rng.permutation(N)
    x, y = x[idx], y[idx]

    x_train, y_train = x[:n_train], y[:n_train]
    x_cal, y_cal = x[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
    x_test, y_test = x[n_train+n_cal:], y[n_train+n_cal:]

    # fit ONCE on the full training set
    model = fit_model(x_train, y_train, degree=15)

    # calibrate ONCE on the full calibration set -> ONE threshold
    cal_resid = np.abs(y_cal - model(x_cal))
    n = len(cal_resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    threshold = np.sort(cal_resid)[k - 1]

    # evaluate on full test set
    y_hat = model(x_test)
    lower, upper = y_hat - threshold, y_hat + threshold
    covered = (y_test >= lower) & (y_test <= upper)
    overall_cov = covered.mean()

    # SPLIT THE TEST SET ONLY (not training or calibration)
    mask_pos = x_test > 0
    mask_neg = x_test < 0

    cov_pos = covered[mask_pos].mean() if mask_pos.sum() > 0 else np.nan
    cov_neg = covered[mask_neg].mean() if mask_neg.sum() > 0 else np.nan

    return overall_cov, cov_pos, cov_neg

# --- run 200 times ---
ALPHA = 0.05
N_RUNS = 200

rng = np.random.default_rng(0)

overall = []
pos = []
neg = []

for i in range(N_RUNS):
    a, b, c = one_run(rng, ALPHA)
    overall.append(a)
    pos.append(b)
    neg.append(c)

overall = np.array(overall)
pos = np.array(pos)
neg = np.array(neg)

print(f"Alpha = {ALPHA}, nominal coverage = {1-ALPHA}")
print(f"Runs = {N_RUNS}\n")
print(f"Overall     mean={overall.mean():.4f}  std={overall.std():.4f}")
print(f"x > 0 half  mean={pos.mean():.4f}  std={pos.std():.4f}")
print(f"x < 0 half  mean={neg.mean():.4f}  std={neg.std():.4f}")
print()
print("The two halves should now differ from each other,")
print("because they share the SAME threshold but have different noise levels.")