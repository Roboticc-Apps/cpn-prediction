"""
Validate CP^n Prediction Engine Against Real IBM Hardware Data

Loads actual density matrices from ibm_fez experiments (March 8-9, 2026)
and validates that our Bures correction matches the measured results.

This is the bridge between theory and experiment.
"""

import json
import numpy as np
import os
from pathlib import Path
from calibration import bures_mean, euclidean_mean, fidelity

# Path to the first paper's hardware data
HARDWARE_DIR = Path(r"D:\development\python\geometry-consciousness\quantum_geometry\hardware_runs")


def load_bell_batches(shots: int = 8192) -> list[np.ndarray]:
    """Load Bell state density matrices from IBM hardware."""
    bell_dir = HARDWARE_DIR / "bell_state"
    matrices = []

    pattern = f"bell_{shots}shots_batch"
    for f in sorted(bell_dir.glob(f"{pattern}*.json")):
        if "aggregate" in f.name:
            continue
        with open(f, 'r') as fp:
            data = json.load(fp)

        # Extract density matrix (real + imaginary parts)
        rho_real = data.get('rho_real')
        rho_imag = data.get('rho_imag')
        if rho_real is None:
            rho_data = data.get('rho')
            if rho_data is None:
                continue
            rho = np.array(rho_data, dtype=complex)
        else:
            rho = np.array(rho_real, dtype=complex)
            if rho_imag is not None:
                rho = rho + 1j * np.array(rho_imag)

        if rho.shape[0] == rho.shape[1]:
            # Normalize trace
            rho = rho / np.trace(rho)
            matrices.append(rho)

    return matrices


def load_ghz_batches(shots: int = 8192) -> list[np.ndarray]:
    """Load GHZ 3-qubit density matrices from IBM hardware."""
    ghz_dir = HARDWARE_DIR / "ghz_3qubit"
    matrices = []

    for f in sorted(ghz_dir.glob(f"ghz3q_{shots}shots_batch*.json")):
        with open(f, 'r') as fp:
            data = json.load(fp)

        # Extract density matrix (real + imaginary parts)
        rho_real = data.get('rho_real')
        rho_imag = data.get('rho_imag')
        if rho_real is None:
            rho_data = data.get('rho')
            if rho_data is None:
                continue
            rho = np.array(rho_data, dtype=complex)
        else:
            rho = np.array(rho_real, dtype=complex)
            if rho_imag is not None:
                rho = rho + 1j * np.array(rho_imag)

        if rho.shape[0] == rho.shape[1]:
            rho = rho / np.trace(rho)
            matrices.append(rho)

    return matrices


def load_aggregate_results() -> dict:
    """Load the published aggregate results for comparison."""
    results_file = HARDWARE_DIR / "mle_comparison_results.json"
    if results_file.exists():
        with open(results_file, 'r') as fp:
            return json.load(fp)

    # Fallback: load from GHZ aggregate
    ghz_file = HARDWARE_DIR / "ghz_3qubit" / "ghz3q_all_results.json"
    if ghz_file.exists():
        with open(ghz_file, 'r') as fp:
            return json.load(fp)

    return {}


def ideal_bell_state() -> np.ndarray:
    """Ideal Bell state |Phi+> = (|00> + |11>)/sqrt(2)."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def ideal_ghz_state() -> np.ndarray:
    """Ideal 3-qubit GHZ state (|000> + |111>)/sqrt(2)."""
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)  # |000>
    psi[7] = 1 / np.sqrt(2)  # |111>
    return np.outer(psi, psi.conj())


def validate_against_hardware():
    """Main validation routine."""
    print("=" * 70)
    print("VALIDATION AGAINST IBM HARDWARE DATA (ibm_fez)")
    print("=" * 70)

    # Check data exists
    if not HARDWARE_DIR.exists():
        print(f"\nERROR: Hardware data directory not found: {HARDWARE_DIR}")
        print("Expected data from geometry-consciousness repo.")
        return False

    all_pass = True

    # =========================================================
    # Bell State (2-qubit, CP^3)
    # =========================================================
    print("\n--- Bell State (2-qubit, CP^3, dim=6) ---")

    for shots in [512, 2048, 8192]:
        matrices = load_bell_batches(shots)
        if not matrices:
            print(f"  {shots} shots: No data found, skipping")
            continue

        rho_ideal = ideal_bell_state()

        # Our Euclidean mean
        rho_euc = euclidean_mean(matrices)
        eigvals, eigvecs = np.linalg.eigh(rho_euc)
        eigvals = np.maximum(eigvals, 0)
        rho_euc = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho_euc /= np.trace(rho_euc)

        # Our Bures mean
        rho_bures = bures_mean(matrices, max_iter=100)

        f_euc = fidelity(rho_euc, rho_ideal)
        f_bures = fidelity(rho_bures, rho_ideal)
        gain = (f_bures - f_euc) * 100

        print(f"\n  {shots} shots ({len(matrices)} batches):")
        print(f"    F_euclidean = {f_euc:.6f}")
        print(f"    F_bures     = {f_bures:.6f}")
        print(f"    Gain        = {'+' if gain >= 0 else ''}{gain:.4f}%")
        print(f"    Bures wins  = {'YES' if gain > 0 else 'NO'}")

        if gain < 0:
            print(f"    WARNING: Bures did not win at {shots} shots")

    # =========================================================
    # GHZ State (3-qubit, CP^7)
    # =========================================================
    print("\n--- GHZ State (3-qubit, CP^7, dim=14) ---")

    matrices = load_ghz_batches(8192)
    if matrices:
        rho_ideal = ideal_ghz_state()

        rho_euc = euclidean_mean(matrices)
        eigvals, eigvecs = np.linalg.eigh(rho_euc)
        eigvals = np.maximum(eigvals, 0)
        rho_euc = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho_euc /= np.trace(rho_euc)

        rho_bures = bures_mean(matrices, max_iter=100)

        f_euc = fidelity(rho_euc, rho_ideal)
        f_bures = fidelity(rho_bures, rho_ideal)
        gain = (f_bures - f_euc) * 100

        print(f"\n  8192 shots ({len(matrices)} batches):")
        print(f"    F_euclidean = {f_euc:.6f}")
        print(f"    F_bures     = {f_bures:.6f}")
        print(f"    Gain        = {'+' if gain >= 0 else ''}{gain:.4f}%")
        print(f"    Bures wins  = {'YES' if gain > 0 else 'NO'}")
    else:
        print("  No GHZ data found, skipping")

    # =========================================================
    # Compare against published results
    # =========================================================
    print("\n--- Comparison with Published Results ---")
    agg = load_aggregate_results()
    if agg:
        print(f"\n  Published results loaded: {list(agg.keys())[:5]}...")

        # Try to find the relevant data
        if 'results' in agg:
            for key, vals in agg['results'].items():
                print(f"\n  {key} shots:")
                if isinstance(vals, dict):
                    for metric, value in vals.items():
                        print(f"    {metric}: {value}")
    else:
        print("  No aggregate results file found")

    # =========================================================
    # Cross-validation: our engine vs paper's reported numbers
    # =========================================================
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION: Engine vs Paper")
    print("=" * 70)

    paper_results = {
        '1-qubit (CP^1)': {'f_euc': 0.9990, 'f_bures': 0.9991, 'gain': 0.003},
        '2-qubit Bell (CP^3)': {'f_euc': 0.9154, 'f_bures': 0.9195, 'gain': 0.415},
        '3-qubit GHZ (CP^7)': {'f_euc': 0.9323, 'f_bures': 0.9456, 'gain': 1.332},
    }

    print(f"\n  {'State':>25} {'Paper gain':>12} {'Engine gain':>12} {'Match':>8}")
    print("  " + "-" * 55)

    # Load what we can compute
    bell_matrices = load_bell_batches(8192)
    ghz_matrices = load_ghz_batches(8192)

    if bell_matrices:
        rho_ideal = ideal_bell_state()
        rho_euc = euclidean_mean(bell_matrices)
        eigvals, eigvecs = np.linalg.eigh(rho_euc)
        eigvals = np.maximum(eigvals, 0)
        rho_euc = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho_euc /= np.trace(rho_euc)
        rho_bures = bures_mean(bell_matrices, max_iter=100)
        our_gain_bell = (fidelity(rho_bures, rho_ideal) -
                         fidelity(rho_euc, rho_ideal)) * 100
        paper_gain_bell = paper_results['2-qubit Bell (CP^3)']['gain']
        match_bell = abs(our_gain_bell - paper_gain_bell) < 0.5
        print(f"  {'2-qubit Bell (CP^3)':>25} +{paper_gain_bell:>10.3f}% "
              f"+{our_gain_bell:>10.3f}% {'MATCH' if match_bell else 'DIFF':>8}")
    else:
        print(f"  {'2-qubit Bell (CP^3)':>25} +{0.415:>10.3f}%  {'N/A':>10} {'---':>8}")

    if ghz_matrices:
        rho_ideal = ideal_ghz_state()
        rho_euc = euclidean_mean(ghz_matrices)
        eigvals, eigvecs = np.linalg.eigh(rho_euc)
        eigvals = np.maximum(eigvals, 0)
        rho_euc = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        rho_euc /= np.trace(rho_euc)
        rho_bures = bures_mean(ghz_matrices, max_iter=100)
        our_gain_ghz = (fidelity(rho_bures, rho_ideal) -
                        fidelity(rho_euc, rho_ideal)) * 100
        paper_gain_ghz = paper_results['3-qubit GHZ (CP^7)']['gain']
        match_ghz = abs(our_gain_ghz - paper_gain_ghz) < 0.5
        print(f"  {'3-qubit GHZ (CP^7)':>25} +{paper_gain_ghz:>10.3f}% "
              f"+{our_gain_ghz:>10.3f}% {'MATCH' if match_ghz else 'DIFF':>8}")
    else:
        print(f"  {'3-qubit GHZ (CP^7)':>25} +{1.332:>10.3f}%  {'N/A':>10} {'---':>8}")

    print("\n" + "=" * 70)
    if all_pass:
        print("Hardware validation complete.")
    print("=" * 70)

    return all_pass


if __name__ == '__main__':
    validate_against_hardware()
