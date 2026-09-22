import numpy as np

rng_global = np.random.default_rng(42)

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

def one_run(rng, alpha, mask_fn, n_total=2000, n_train=30, n_cal=500):
    # generate fresh data
    x, y = make_data(n_total, rng)

    # apply subset mask (x>0, x<0, or all)
    keep = mask_fn(x)
    x, y = x[keep], y[keep]

    # shuffle
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]

    # split
    if len(x) < n_train + n_cal + 100:
        return None  # too small, skip
    x_train, y_train = x[:n_train], y[:n_train]
    x_cal, y_cal = x[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
    x_test, y_test = x[n_train+n_cal:], y[n_train+n_cal:]

    # fit
    model = fit_model(x_train, y_train, degree=15)

    # calibration residuals
    cal_resid = np.abs(y_cal - model(x_cal))
    n = len(cal_resid)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    threshold = np.sort(cal_resid)[k - 1]

    # test coverage
    y_hat = model(x_test)
    covered = (y_test >= y_hat - threshold) & (y_test <= y_hat + threshold)
    return covered.mean()

def replicate(n_runs, alpha, mask_fn, label):
    rng = np.random.default_rng(0)
    results = []
    for _ in range(n_runs):
        cov = one_run(rng, alpha, mask_fn)
        if cov is not None:
            results.append(cov)
    results = np.array(results)
    print(f"{label:15s} runs={len(results):3d}  mean={results.mean():.4f}  "
          f"std={results.std():.4f}  min={results.min():.4f}  max={results.max():.4f}")
    return results

# --- main ---
ALPHA = 0.05
N_RUNS = 1000

print(f"Alpha = {ALPHA}, nominal coverage = {1-ALPHA}")
print(f"Runs per subset = {N_RUNS}, total = {N_RUNS * 3}\n")

r_pos = replicate(N_RUNS, ALPHA, lambda x: x > 0, "x>0")
r_neg = replicate(N_RUNS, ALPHA, lambda x: x < 0, "x<0")
r_all = replicate(N_RUNS, ALPHA, lambda x: np.ones_like(x, dtype=bool), "combined")

print()
print("=== Summary ===")
for label, r in [("x>0", r_pos), ("x<0", r_neg), ("combined", r_all)]:
    ok = "✓ in [0.90, 0.95]" if 0.90 <= r.mean() <= 0.95 else "✗ outside [0.90, 0.95]"
    print(f"{label:10s} mean={r.mean():.4f}  {ok}")

overall = np.concatenate([r_pos, r_neg, r_all])
print(f"\nOverall mean across all 600 runs: {overall.mean():.4f}")
print(f"In [0.90, 0.95]: {'YES' if 0.90 <= overall.mean() <= 0.95 else 'NO'}")