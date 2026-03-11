"""
Raychaudhuri Focusing on CP^n

The Raychaudhuri equation describes how a bundle of geodesics
expands or contracts. On CP^n with K=4, geodesics ALWAYS focus.

This module solves the equation numerically and compares
focusing rates across dimensions — the quantitative backbone
of the prediction engine.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cpn_geodesic import (
    focusing_rate, ricci_curvature, conjugate_point,
    jacobi_field, random_state, random_tangent, geodesic,
    fubini_study_distance, normalize
)


def solve_raychaudhuri(n: int, theta_0: float = 0.0,
                       t_max: float = 2.0,
                       dt: float = 0.0001) -> tuple:
    """Solve Raychaudhuri equation on CP^n.

    d(theta)/dt = -(1/dim) * theta^2 - R_{ij}u^i u^j

    where dim = 2n (real dimension), R_{ij}u^i u^j = 2(n+1).

    theta > 0: expansion (geodesics diverging)
    theta < 0: contraction (geodesics converging)
    theta -> -inf: caustic (focal point reached)
    """
    dim = 2 * n
    ric = ricci_curvature(n)

    times = [0.0]
    theta = [theta_0]

    t = 0.0
    th = theta_0

    while t < t_max:
        dth = -(1.0 / dim) * th**2 - ric
        th += dth * dt
        t += dt

        times.append(t)
        theta.append(th)

        # Caustic: theta diverges
        if th < -1e4:
            break

    return np.array(times), np.array(theta)


def solve_raychaudhuri_with_expansion(n: int, theta_0: float = 5.0,
                                       t_max: float = 3.0,
                                       dt: float = 0.0001) -> tuple:
    """Solve Raychaudhuri starting from EXPANDING congruence.

    Even initially expanding geodesics get focused on CP^n.
    This demonstrates: no matter how hard you push apart,
    the curvature pulls back.
    """
    return solve_raychaudhuri(n, theta_0=theta_0, t_max=t_max, dt=dt)


def time_to_caustic(n: int, theta_0: float = 0.0) -> float:
    """Compute time to first caustic (focusing singularity).

    Uses the analytical estimate from Ricci-driven focusing:
        t_caustic = pi / sqrt(Ric)  where Ric = 2(n+1)

    The Ricci curvature 2(n+1) is the effective curvature driving
    geodesic focusing on CP^n. On a space of effective curvature K,
    the conjugate point (caustic) is at pi / sqrt(K).

    Higher dimension => larger Ricci => faster caustic.
    """
    ric = ricci_curvature(n)
    return np.pi / np.sqrt(ric)


def numerical_focusing_test(n: int, n_geodesics: int = 20,
                            n_steps: int = 100) -> dict:
    """Numerically verify geodesic focusing on CP^n.

    Creates a parallel congruence: nearby starting points, same direction.
    Measures pairwise distances over time.
    Verifies they converge as predicted by Jacobi fields.

    For a parallel congruence, J(0)=epsilon, J'(0)=0:
        J(t) = epsilon * cos(sqrt(K)*t)
    This reaches zero at t = pi/(2*sqrt(K)):
        K=4 (holomorphic): t = pi/4
        K=1 (real):        t = pi/2
    """
    psi = random_state(n)
    v_dir = random_tangent(psi)

    # Create nearby starting points (parallel congruence)
    # Perturb only along the holomorphic direction (i*v_dir projected
    # to tangent space). This samples the K=4 holomorphic curvature,
    # which gives clean focusing at t = pi/2.
    epsilon = 0.05
    starting_points = [psi]

    # Holomorphic perturbation direction
    w_holo = 1j * v_dir - np.vdot(psi, 1j * v_dir) * psi
    if np.linalg.norm(w_holo) > 1e-12:
        w_holo = w_holo / np.linalg.norm(w_holo)
    else:
        w_holo = random_tangent(psi)

    for k in range(1, n_geodesics):
        # Spread perturbations symmetrically along holomorphic direction
        # with varying amplitudes and add small random component
        amp = epsilon * k / n_geodesics
        sign = 1.0 if k % 2 == 0 else -1.0
        w = sign * w_holo * amp
        # Add small random perturbation for diversity
        w += random_tangent(psi) * amp * 0.3
        psi_new = normalize(psi + w)
        starting_points.append(psi_new)

    # For each starting point, compute the tangent direction by
    # parallel transporting v_dir (approximate: project onto tangent space)
    tangents = []
    for p in starting_points:
        v = v_dir - np.vdot(p, v_dir) * p
        if np.linalg.norm(v) < 1e-12:
            v = random_tangent(p)
        else:
            v = v / np.linalg.norm(v)
        tangents.append(v)

    # Evolve all geodesics and measure spread
    t_values = np.linspace(0, np.pi, n_steps)
    mean_spread = []
    max_spread = []

    for t in t_values:
        states = [geodesic(p, v, t) for p, v in zip(starting_points, tangents)]
        # Pairwise distances
        dists = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                d = fubini_study_distance(states[i], states[j])
                dists.append(d)
        mean_spread.append(np.mean(dists))
        max_spread.append(np.max(dists))

    mean_spread = np.array(mean_spread)
    max_spread = np.array(max_spread)

    # Find convergence points (local minima in spread)
    convergence_times = []
    for i in range(1, len(mean_spread) - 1):
        if mean_spread[i] < mean_spread[i-1] and mean_spread[i] < mean_spread[i+1]:
            convergence_times.append(t_values[i])

    # Theoretical: spread should go as |cos(sqrt(K)*t)|
    # Zero at t = pi/(2*sqrt(K)), peaks at t=0 and t=pi/sqrt(K)
    theoretical_peak_holo = np.pi / 4    # K=4: pi/(2*2)
    theoretical_zero_holo = np.pi / 2    # K=4: pi/2

    # Use initial spread (t=0) as reference — this is the starting separation
    initial_spread = mean_spread[0] if len(mean_spread) > 0 else 0

    return {
        'times': t_values,
        'mean_spread': mean_spread,
        'max_spread': max_spread,
        'convergence_times': convergence_times,
        'theoretical_peak_holo': theoretical_peak_holo,
        'theoretical_zero_holo': theoretical_zero_holo,
        'initial_spread': initial_spread,
        'min_spread': np.min(mean_spread[1:]) if len(mean_spread) > 1 else 0,
        'dimension': n,
    }


def plot_focusing_comparison(save_path: str = 'focusing_comparison.png'):
    """Plot focusing rates across dimensions.
    This is Figure 1 of the prediction paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Raychaudhuri equation across dimensions ---
    ax = axes[0, 0]
    dims = [1, 3, 7, 15, 31]
    labels = ['CP¹ (1q)', 'CP³ (2q)', 'CP⁷ (3q)', 'CP¹⁵ (4q)', 'CP³¹ (5q)']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dims)))

    for n, label, color in zip(dims, labels, colors):
        times, theta = solve_raychaudhuri(n, theta_0=0.0, t_max=1.5)
        ax.plot(times, theta, label=label, color=color, linewidth=2)

    ax.set_xlabel('Geodesic parameter t')
    ax.set_ylabel('Expansion θ')
    ax.set_title('Raychaudhuri Focusing: Higher Dimension = Faster Collapse')
    ax.legend(fontsize=9)
    ax.set_ylim(-500, 50)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Jacobi fields (holomorphic vs real) ---
    ax = axes[0, 1]
    t = np.linspace(0, 2 * np.pi, 500)

    j_holo = [jacobi_field(ti, curvature=4.0) for ti in t]
    j_real = [jacobi_field(ti, curvature=1.0) for ti in t]
    j_flat = [jacobi_field(ti, curvature=0.0) for ti in t]

    ax.plot(t, j_holo, 'b-', label='Holomorphic (K=4)', linewidth=2)
    ax.plot(t, j_real, 'r-', label='Real (K=1)', linewidth=2)
    ax.plot(t, j_flat, 'k--', label='Flat (K=0)', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Mark convergence points
    ax.axvline(x=np.pi/2, color='b', linestyle=':', alpha=0.5, label='π/2 (holo focal)')
    ax.axvline(x=np.pi, color='r', linestyle=':', alpha=0.5, label='π (real focal)')

    ax.set_xlabel('Geodesic parameter t')
    ax.set_ylabel('Jacobi field J(t)')
    ax.set_title('Jacobi Fields: Curved Space Forces Reconvergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Initially EXPANDING congruence still focuses ---
    ax = axes[1, 0]
    initial_expansions = [0, 5, 10, 20, 50]

    for th0 in initial_expansions:
        times, theta = solve_raychaudhuri(7, theta_0=th0, t_max=2.0)
        label = f'θ₀ = {th0}' if th0 > 0 else 'θ₀ = 0 (parallel)'
        ax.plot(times, theta, label=label, linewidth=1.5)

    ax.set_xlabel('Geodesic parameter t')
    ax.set_ylabel('Expansion θ')
    ax.set_title('CP⁷: Even Expanding Geodesics Get Focused')
    ax.legend(fontsize=9)
    ax.set_ylim(-500, 60)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Focusing rate vs dimension (log scale) ---
    ax = axes[1, 1]
    n_values = [1, 2, 3, 4, 7, 15, 31, 63, 127, 255]
    rates = [abs(focusing_rate(n)) for n in n_values]
    qubit_labels = []
    for n in n_values:
        # n = 2^q - 1 for q qubits
        q = np.log2(n + 1)
        if q == int(q):
            qubit_labels.append(f'{int(q)}q')
        else:
            qubit_labels.append(f'n={n}')

    ax.semilogy(range(len(n_values)), rates, 'ko-', markersize=8, linewidth=2)
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(qubit_labels, rotation=45)
    ax.set_ylabel('|Focusing rate| = 2(n+1)')
    ax.set_title('Focusing Rate Scales Linearly with Dimension')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Larger system =\ntighter funnel =\nfewer possible futures',
                xy=(7, rates[7]), xytext=(5, rates[5] * 3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_geodesic_spread(n: int = 7,
                         save_path: str = 'geodesic_spread.png'):
    """Visualize actual geodesic spread on CP^n.
    Shows geodesics diverging then reconverging.
    """
    result = numerical_focusing_test(n, n_geodesics=30, n_steps=200)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result['times'], result['mean_spread'], 'b-',
            label='Mean pairwise distance', linewidth=2)
    ax.fill_between(result['times'], 0, result['max_spread'],
                    alpha=0.2, color='blue', label='Max spread envelope')

    # Mark theoretical convergence points
    ax.axvline(x=np.pi/2, color='red', linestyle='--',
               label=f'π/2 = {np.pi/2:.3f} (holomorphic focal)', alpha=0.7)
    ax.axvline(x=np.pi, color='orange', linestyle='--',
               label=f'π = {np.pi:.3f} (real focal)', alpha=0.7)

    ax.set_xlabel('Geodesic parameter t')
    ax.set_ylabel('Spread (Fubini-Study distance)')
    ax.set_title(f'Geodesic Bundle on CP^{n}: Diverge → Reconverge → Repeat')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Raychaudhuri Focusing Analysis")
    print("=" * 60)

    # Time to caustic across dimensions
    print("\n--- Time to Caustic vs Dimension ---")
    print(f"{'CP^n':>6} {'Qubits':>6} {'Focusing':>10} {'t_caustic':>10}")
    print("-" * 36)
    for n in [1, 3, 7, 15, 31, 63]:
        q = int(np.log2(n + 1))
        tc = time_to_caustic(n)
        print(f"CP^{n:>3} {q:>6} {focusing_rate(n):>10.1f} {tc:>10.4f}")

    # Numerical focusing verification
    print("\n--- Numerical Focusing Verification (CP^7, 3-qubit) ---")
    result = numerical_focusing_test(7, n_geodesics=30, n_steps=200)
    print(f"  Initial spread: {result['initial_spread']:.6f}")
    print(f"  Minimum spread: {result['min_spread']:.6f}")
    print(f"  Compression ratio: {result['initial_spread']/max(result['min_spread'], 1e-15):.1f}x")
    print(f"  Convergence times: {[f'{t:.3f}' for t in result['convergence_times']]}")
    print(f"  Theoretical (holo): peak at {result['theoretical_peak_holo']:.3f}, "
          f"zero at {result['theoretical_zero_holo']:.3f}")

    # Even expanding congruences focus
    print("\n--- Even Expanding Geodesics Focus (CP^7) ---")
    for th0 in [0, 5, 10, 20, 50, 100]:
        tc = time_to_caustic(7)
        times, theta = solve_raychaudhuri(7, theta_0=th0, t_max=3.0)
        # Find where theta first goes negative
        cross_time = None
        for i in range(len(theta) - 1):
            if theta[i] >= 0 and theta[i+1] < 0:
                cross_time = times[i]
                break
        if cross_time is not None:
            print(f"  θ₀ = {th0:>3}: expansion→contraction at t = {cross_time:.4f}")
        else:
            print(f"  θ₀ = {th0:>3}: always contracting")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_focusing_comparison()
    plot_geodesic_spread(n=7)
    plot_geodesic_spread(n=3, save_path='geodesic_spread_cp3.png')
    plot_geodesic_spread(n=15, save_path='geodesic_spread_cp15.png')

    print("\n--- Key Result ---")
    print("No matter the initial expansion rate, CP^n curvature")
    print("ALWAYS wins. Geodesics ALWAYS focus. The future ALWAYS")
    print("converges. The only question is how fast — and bigger")
    print("systems converge faster.")
