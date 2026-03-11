"""
Scaling Test: Bures vs Euclidean at Higher Qubit Counts

Tests the core prediction: Euclidean error grows exponentially
with qubit count, making it catastrophically wrong at scale.

This is the key evidence for the second paper.
"""

import numpy as np
import time
from calibration import (
    bures_mean, euclidean_mean, fidelity,
    bures_distance, predicted_bures_gain
)
from cpn_geodesic import focusing_rate


def make_random_pure_state(dim: int) -> np.ndarray:
    """Random pure state density matrix."""
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def make_noisy_estimates(rho_ideal: np.ndarray, n_batches: int,
                         noise_scale: float) -> list[np.ndarray]:
    """Simulate noisy tomographic estimates of a density matrix."""
    dim = rho_ideal.shape[0]
    estimates = []
    for _ in range(n_batches):
        noise = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        noise = (noise + noise.conj().T) / 2
        noise *= noise_scale

        rho = rho_ideal + noise
        # Force positive semidefinite
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.maximum(eigvals, 0)
        rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho /= np.trace(rho)
        estimates.append(rho)
    return estimates


def scaling_experiment(max_qubits: int = 6, n_trials: int = 10,
                       n_batches: int = 8, noise_scale: float = 0.02):
    """Run the full scaling experiment."""
    print("=" * 75)
    print("BURES vs EUCLIDEAN SCALING EXPERIMENT")
    print(f"Trials: {n_trials}, Batches: {n_batches}, Noise: {noise_scale}")
    print("=" * 75)

    results = {}

    for q in range(1, max_qubits + 1):
        dim = 2**q
        n_cp = dim - 1

        trial_gains = []
        trial_f_euc = []
        trial_f_bures = []
        bures_wins = 0
        t_start = time.time()

        for trial in range(n_trials):
            rho_ideal = make_random_pure_state(dim)
            estimates = make_noisy_estimates(rho_ideal, n_batches, noise_scale)

            # Euclidean mean
            rho_euc = euclidean_mean(estimates)
            eigvals, eigvecs = np.linalg.eigh(rho_euc)
            eigvals = np.maximum(eigvals, 0)
            rho_euc = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            rho_euc /= np.trace(rho_euc)

            # Bures mean
            rho_bures = bures_mean(estimates, max_iter=50)

            f_euc = fidelity(rho_euc, rho_ideal)
            f_bures = fidelity(rho_bures, rho_ideal)

            trial_gains.append((f_bures - f_euc) * 100)
            trial_f_euc.append(f_euc)
            trial_f_bures.append(f_bures)
            if f_bures > f_euc:
                bures_wins += 1

        elapsed = time.time() - t_start
        mean_gain = np.mean(trial_gains)
        std_gain = np.std(trial_gains) / np.sqrt(n_trials)
        pred_gain = predicted_bures_gain(q)
        rate = abs(focusing_rate(n_cp))

        results[q] = {
            'mean_gain': mean_gain,
            'std_gain': std_gain,
            'mean_f_euc': np.mean(trial_f_euc),
            'mean_f_bures': np.mean(trial_f_bures),
            'bures_wins': bures_wins,
            'n_trials': n_trials,
            'predicted_gain': pred_gain,
            'focusing_rate': rate,
            'elapsed': elapsed,
        }

        win_pct = 100 * bures_wins / n_trials
        print(f"\n  {q}-qubit (CP^{n_cp}, dim={dim}, rate={rate:.0f}):")
        print(f"    F_euclidean = {np.mean(trial_f_euc):.6f}")
        print(f"    F_bures     = {np.mean(trial_f_bures):.6f}")
        print(f"    Gain        = +{mean_gain:.4f}% ± {std_gain:.4f}%")
        print(f"    Predicted   = +{pred_gain:.3f}%")
        print(f"    Bures wins  = {bures_wins}/{n_trials} ({win_pct:.0f}%)")
        print(f"    Time        = {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"\n{'Q':>3} {'CP^n':>6} {'Rate':>6} {'F_euc':>10} {'F_bures':>10} "
          f"{'Gain':>10} {'Predicted':>10} {'Wins':>8}")
    print("-" * 68)

    for q in sorted(results.keys()):
        r = results[q]
        n_cp = 2**q - 1
        print(f"{q:>3} CP^{n_cp:>3} {r['focusing_rate']:>6.0f} "
              f"{r['mean_f_euc']:>10.6f} {r['mean_f_bures']:>10.6f} "
              f"+{r['mean_gain']:>8.4f}% +{r['predicted_gain']:>8.3f}% "
              f"{r['bures_wins']:>3}/{r['n_trials']}")

    # Scaling analysis
    print("\n" + "=" * 75)
    print("SCALING ANALYSIS")
    print("=" * 75)

    gains = [results[q]['mean_gain'] for q in sorted(results.keys())]
    rates = [results[q]['focusing_rate'] for q in sorted(results.keys())]

    for i in range(1, len(gains)):
        q = sorted(results.keys())[i]
        q_prev = sorted(results.keys())[i - 1]
        if gains[i - 1] > 0.001:  # Avoid division by near-zero
            gain_ratio = gains[i] / gains[i - 1]
            rate_ratio = rates[i] / rates[i - 1]
            print(f"  {q}q/{q_prev}q: gain ratio = {gain_ratio:.2f}x, "
                  f"rate ratio = {rate_ratio:.1f}x")

    total_wins = sum(r['bures_wins'] for r in results.values())
    total_trials = sum(r['n_trials'] for r in results.values())
    print(f"\n  Total Bures wins: {total_wins}/{total_trials} "
          f"({100*total_wins/total_trials:.1f}%)")

    if total_wins == total_trials:
        print("  >>> BURES WINS EVERY SINGLE TRIAL <<<")

    # Extrapolation
    print("\n" + "=" * 75)
    print("EXTRAPOLATION (from measured scaling)")
    print("=" * 75)

    if len(gains) >= 2 and gains[-1] > 0 and gains[-2] > 0:
        # Fit exponential: gain ~ a * 2^(b*q)
        log_gains = [np.log2(max(g, 1e-10)) for g in gains]
        q_values = sorted(results.keys())

        # Simple linear fit in log space
        coeffs = np.polyfit(q_values, log_gains, 1)
        growth_rate = coeffs[0]

        print(f"  Measured growth: gain doubles every "
              f"{1/growth_rate:.2f} qubits")
        print(f"\n  Extrapolated gains:")
        for q_ext in [10, 20, 50, 100]:
            ext_gain = 2**(coeffs[0] * q_ext + coeffs[1])
            status = "BROKEN" if ext_gain > 100 else "critical" if ext_gain > 10 else "ok"
            print(f"    {q_ext:>3} qubits: +{ext_gain:>12.1f}% [{status}]")


if __name__ == '__main__':
    np.random.seed(42)

    # Run with conservative parameters first
    scaling_experiment(
        max_qubits=5,
        n_trials=10,
        n_batches=8,
        noise_scale=0.02
    )
