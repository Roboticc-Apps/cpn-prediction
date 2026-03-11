"""
CP^n Geodesic Prediction Engine

Core geodesic computation on complex projective space CP^n
with Fubini-Study metric (holomorphic sectional curvature K=4).

Geodesics on CP^n are projections of great circles from S^{2n+1}.
The curvature K=4 forces geodesic focusing — nearby trajectories
reconverge at known distances. This is the mathematical basis
for geometric prediction.
"""

import numpy as np
from typing import Optional


# =============================================================
# State Space: CP^n
# =============================================================

def normalize(psi: np.ndarray) -> np.ndarray:
    """Normalize a state vector to unit norm."""
    return psi / np.linalg.norm(psi)


def remove_phase(psi: np.ndarray) -> np.ndarray:
    """Fix global phase so first nonzero component is real positive.
    States on CP^n are equivalence classes [psi] = {e^{i*alpha} * psi}.
    """
    psi = normalize(psi)
    # Find first component with magnitude > threshold
    for i in range(len(psi)):
        if abs(psi[i]) > 1e-10:
            phase = np.exp(-1j * np.angle(psi[i]))
            return psi * phase
    return psi


def random_state(n: int) -> np.ndarray:
    """Random state on CP^n (uniform Haar measure).
    Returns unit vector in C^{n+1}.
    """
    psi = np.random.randn(n + 1) + 1j * np.random.randn(n + 1)
    return normalize(psi)


def random_tangent(psi: np.ndarray) -> np.ndarray:
    """Random unit tangent vector at psi on CP^n.
    Must satisfy: <v|psi> = 0 (orthogonal to state).
    """
    n = len(psi)
    v = np.random.randn(n) + 1j * np.random.randn(n)
    # Project out component along psi
    v = v - np.vdot(psi, v) * psi
    if np.linalg.norm(v) < 1e-12:
        return random_tangent(psi)  # Retry on degenerate case
    return v / np.linalg.norm(v)


# =============================================================
# Fubini-Study Metric
# =============================================================

def fubini_study_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    """Fubini-Study distance on CP^n.
    d_FS = arccos|<psi|phi>|
    Range: [0, pi/2]. Maximum at pi/2 (orthogonal states).
    """
    psi, phi = normalize(psi), normalize(phi)
    overlap = abs(np.vdot(psi, phi))
    # Clamp for numerical stability
    overlap = min(overlap, 1.0)
    return np.arccos(overlap)


def fubini_study_inner_product(v: np.ndarray, w: np.ndarray,
                                psi: np.ndarray) -> complex:
    """Fubini-Study Hermitian inner product on T_{[psi]}CP^n.
    g_FS(v, w) = Re(<v|w> - <v|psi><psi|w>)
    """
    return np.real(np.vdot(v, w) - np.vdot(v, psi) * np.vdot(psi, w))


# =============================================================
# Geodesics on CP^n
# =============================================================

def geodesic(psi: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    """Compute the geodesic gamma(t) on CP^n.

    gamma(t) = cos(t)|psi> + sin(t)|v>

    where |psi> is the starting point (unit vector) and
    |v> is the unit tangent vector (orthogonal to |psi>).

    This is a great circle on S^{2n+1} projected to CP^n.
    Period: pi (returns to same point on CP^n due to phase identification).
    """
    psi = normalize(psi)
    # Ensure v is orthogonal to psi and unit
    v = v - np.vdot(psi, v) * psi
    v = v / np.linalg.norm(v)
    return normalize(np.cos(t) * psi + np.sin(t) * v)


def geodesic_trajectory(psi: np.ndarray, v: np.ndarray,
                        t_max: float = np.pi,
                        n_steps: int = 100) -> list[np.ndarray]:
    """Compute a sequence of states along a geodesic."""
    times = np.linspace(0, t_max, n_steps)
    return [geodesic(psi, v, t) for t in times], times


def geodesic_between(psi: np.ndarray, phi: np.ndarray,
                     t: float) -> np.ndarray:
    """Geodesic from psi toward phi, evaluated at parameter t.
    t=0 gives psi, t=d_FS(psi,phi) gives phi.
    """
    psi, phi = normalize(psi), normalize(phi)
    d = fubini_study_distance(psi, phi)
    if d < 1e-12:
        return psi

    # Tangent vector pointing from psi toward phi
    # phi = cos(d)*psi + sin(d)*v  =>  v = (phi - cos(d)*psi) / sin(d)
    phase = np.vdot(psi, phi) / abs(np.vdot(psi, phi))  # Align phases
    phi_aligned = phi / phase
    v = (phi_aligned - np.cos(d) * psi) / np.sin(d)
    v = normalize(v)

    return geodesic(psi, v, t)


# =============================================================
# Curvature
# =============================================================

def sectional_curvature(psi: np.ndarray, v: np.ndarray,
                        w: np.ndarray) -> float:
    """Sectional curvature of the 2-plane spanned by v, w at psi.

    On CP^n with Fubini-Study metric:
        K(v, w) = 1 + 3*cos^2(alpha)

    where alpha is the angle between the 2-plane and the
    holomorphic direction. Range: [1, 4].

    K = 4: holomorphic plane (maximum curvature)
    K = 1: totally real plane (minimum curvature)
    """
    psi = normalize(psi)
    # Project v, w to tangent space of CP^n at psi
    v = v - np.vdot(psi, v) * psi
    w = w - np.vdot(psi, w) * psi

    if np.linalg.norm(v) < 1e-12 or np.linalg.norm(w) < 1e-12:
        return 4.0  # Degenerate, return max

    v = v / np.linalg.norm(v)
    w = w / np.linalg.norm(w)

    # The Kahler angle: cos(alpha) = |<v|Jw>| / (|v||w|)
    # On CP^n, J acts as multiplication by i in the tangent space
    # The holomorphic sectional curvature formula:
    # K(v,w) = 1 + 3 * |<v|iw> - <v|psi><psi|iw>|^2 / (area^2)

    # Compute the area of the parallelogram
    gvv = np.real(np.vdot(v, v) - abs(np.vdot(v, psi))**2)
    gww = np.real(np.vdot(w, w) - abs(np.vdot(w, psi))**2)
    gvw = np.vdot(v, w) - np.vdot(v, psi) * np.vdot(psi, w)
    area_sq = gvv * gww - abs(gvw)**2

    if area_sq < 1e-20:
        return 4.0

    # Kahler form: omega(v, w) = Im(<v|w> - <v|psi><psi|w>)
    omega = np.imag(np.vdot(v, w) - np.vdot(v, psi) * np.vdot(psi, w))

    cos_alpha_sq = omega**2 / area_sq
    return 1.0 + 3.0 * cos_alpha_sq


def ricci_curvature(n: int) -> float:
    """Ricci curvature of CP^n with Fubini-Study metric.
    Ric = 2(n+1) * g_FS

    Returns the Ricci scalar in the direction of any unit tangent vector.
    This is constant on CP^n (Einstein manifold).
    """
    return 2.0 * (n + 1)


def scalar_curvature(n: int) -> float:
    """Scalar curvature of CP^n.
    R = 4n(n+1)
    """
    return 4.0 * n * (n + 1)


# =============================================================
# Jacobi Fields — How Nearby Geodesics Converge/Diverge
# =============================================================

def jacobi_field(t: float, curvature: float,
                 j0: float = 0.0, dj0: float = 1.0) -> float:
    """Jacobi field magnitude along a geodesic with given sectional curvature.

    J'' + K * J = 0

    Solutions:
        K > 0: J(t) = j0*cos(sqrt(K)*t) + dj0*sin(sqrt(K)*t)/sqrt(K)
        K = 0: J(t) = j0 + dj0*t
        K < 0: J(t) = j0*cosh(sqrt(-K)*t) + dj0*sinh(sqrt(-K)*t)/sqrt(-K)

    For CP^n: K is between 1 and 4 (always positive).
    Positive K => oscillatory => FOCUSING. Geodesics reconverge.

    Default initial conditions: J(0) = 0, J'(0) = 1
    (geodesics starting from same point, initially diverging)
    """
    if curvature > 0:
        sqrt_k = np.sqrt(curvature)
        return j0 * np.cos(sqrt_k * t) + dj0 * np.sin(sqrt_k * t) / sqrt_k
    elif curvature < 0:
        sqrt_k = np.sqrt(-curvature)
        return j0 * np.cosh(sqrt_k * t) + dj0 * np.sinh(sqrt_k * t) / sqrt_k
    else:
        return j0 + dj0 * t


def conjugate_point(curvature: float) -> float:
    """Distance to first conjugate point (where geodesics reconverge).

    For curvature K > 0: t_conjugate = pi / sqrt(K)

    On CP^n:
        Holomorphic direction (K=4): t = pi/2
        Real direction (K=1): t = pi

    Returns float('inf') for K <= 0 (no conjugate point).
    """
    if curvature <= 0:
        return float('inf')
    return np.pi / np.sqrt(curvature)


def focusing_rate(n: int) -> float:
    """Rate of geodesic focusing on CP^n.

    From Raychaudhuri equation with zero initial expansion:
        d(theta)/dt |_{t=0} = -Ric(u, u) = -2(n+1)

    Higher dimension = faster focusing = more constrained future.

    This is the KEY quantitative prediction:
    larger systems have stronger geometric convergence.
    """
    return -2.0 * (n + 1)


# =============================================================
# Raychaudhuri Equation — Evolution of Geodesic Congruence
# =============================================================

def raychaudhuri_evolution(n: int, t_max: float = np.pi / 2,
                           n_steps: int = 200,
                           theta_0: float = 0.0) -> tuple:
    """Solve the Raychaudhuri equation on CP^n.

    d(theta)/dt = -(1/(2n)) * theta^2 - sigma^2 + omega^2 - R_{ij}u^i u^j

    For a geodesic congruence (sigma = omega = 0 initially):
        d(theta)/dt = -(1/(2n)) * theta^2 - 2(n+1)

    theta < 0 means focusing (convergence).
    theta -> -inf means caustic (focal point).

    Returns (times, theta_values, expansion_rates).
    """
    ric = 2.0 * (n + 1)
    dim = 2 * n  # Real dimension of CP^n

    dt = t_max / n_steps
    times = np.zeros(n_steps + 1)
    theta = np.zeros(n_steps + 1)
    theta[0] = theta_0

    for i in range(n_steps):
        times[i + 1] = times[i] + dt
        # Raychaudhuri: d(theta)/dt = -(1/dim)*theta^2 - ric
        dtheta = -(1.0 / dim) * theta[i]**2 - ric
        theta[i + 1] = theta[i] + dtheta * dt

        # Stop if theta diverges (caustic reached)
        if theta[i + 1] < -1e6:
            times = times[:i + 2]
            theta = theta[:i + 2]
            break

    return times, theta


# =============================================================
# Attractor Basin Analysis
# =============================================================

def find_focal_points(psi: np.ndarray, n_directions: int = 50) -> dict:
    """Find focal points from a state on CP^n.

    Shoots geodesics in random tangent directions and finds
    where they reconverge (conjugate points).

    On CP^n with K=4:
    - ALL geodesics reconverge (positive curvature guarantees this)
    - Holomorphic geodesics reconverge at t = pi/2
    - Real geodesics reconverge at t = pi
    - The cut locus is a CP^{n-1} at distance pi/2

    Returns dict with focal point statistics.
    """
    n = len(psi) - 1  # CP^n dimension
    psi = normalize(psi)

    focal_distances = []
    focal_states = []

    for _ in range(n_directions):
        v = random_tangent(psi)

        # Compute sectional curvatures for random planes through this direction
        w = random_tangent(psi)
        K = sectional_curvature(psi, v, w)

        # Conjugate point distance for this curvature
        t_conj = conjugate_point(K)
        focal_distances.append(t_conj)

        # State at the focal point
        focal_state = geodesic(psi, v, t_conj)
        focal_states.append(focal_state)

    focal_distances = np.array(focal_distances)

    return {
        'min_focal_distance': np.min(focal_distances),
        'max_focal_distance': np.max(focal_distances),
        'mean_focal_distance': np.mean(focal_distances),
        'cut_locus_distance': np.pi / 2,  # Always pi/2 on CP^n
        'n_directions': n_directions,
        'dimension': n,
        'focusing_rate': focusing_rate(n),
        'focal_distances': focal_distances,
    }


def convergence_prediction(states: list[np.ndarray],
                           n_future: int = 20) -> dict:
    """Given a sequence of observed states, predict convergence.

    Method:
    1. Fit the trajectory to a geodesic on CP^n
    2. Extrapolate forward along the geodesic
    3. Compute Jacobi field bounds (convergence envelope)
    4. Return predicted future states with confidence

    The Jacobi fields SHRINK on positively curved CP^n,
    meaning prediction confidence INCREASES with time
    (up to the focal point, then it resets).
    """
    if len(states) < 2:
        raise ValueError("Need at least 2 states to predict")

    # Normalize all states
    states = [normalize(s) for s in states]
    n = len(states[0]) - 1  # CP^n dimension

    # Fit: use first and last state to define geodesic direction
    psi_start = states[0]
    psi_end = states[-1]
    d_total = fubini_study_distance(psi_start, psi_end)

    if d_total < 1e-12:
        # States haven't moved — return current state
        return {
            'predicted_states': [psi_start] * n_future,
            'confidence': [1.0] * n_future,
            'geodesic_direction': np.zeros_like(psi_start),
            'speed': 0.0,
        }

    # Extract tangent direction from start toward end
    phase = np.vdot(psi_start, psi_end) / abs(np.vdot(psi_start, psi_end))
    psi_end_aligned = psi_end / phase
    v = (psi_end_aligned - np.cos(d_total) * psi_start) / np.sin(d_total)
    v = normalize(v)

    # Estimate speed (distance per step)
    n_observed = len(states)
    speed = d_total / (n_observed - 1)

    # Measure fit quality: how well do intermediate states lie on geodesic?
    residuals = []
    for i, state in enumerate(states):
        t_expected = speed * i
        predicted = geodesic(psi_start, v, t_expected)
        residual = fubini_study_distance(state, predicted)
        residuals.append(residual)
    fit_quality = 1.0 - np.mean(residuals) / (np.pi / 2)

    # Predict future states
    predicted_states = []
    confidence = []
    for i in range(n_future):
        t = d_total + speed * (i + 1)
        pred = geodesic(psi_start, v, t)
        predicted_states.append(pred)

        # Jacobi field gives convergence envelope
        # On CP^n, the envelope oscillates with period related to curvature
        # Average curvature for confidence: use K=4 (holomorphic, tightest)
        j_holo = abs(jacobi_field(t, curvature=4.0))
        j_real = abs(jacobi_field(t, curvature=1.0))
        j_avg = (j_holo + j_real) / 2

        # Confidence: high when Jacobi field is small (near focal point)
        # Low when Jacobi field is large (between focal points)
        # Normalized so max possible divergence = 0 confidence
        max_j = 1.0 / np.sqrt(1.0)  # max for K=1
        conf = max(0.0, 1.0 - j_avg / max_j) * fit_quality
        confidence.append(conf)

    return {
        'predicted_states': predicted_states,
        'confidence': confidence,
        'geodesic_direction': v,
        'speed': speed,
        'fit_quality': fit_quality,
        'residuals': residuals,
        'focusing_rate': focusing_rate(n),
    }


# =============================================================
# Key Insight Functions
# =============================================================

def max_future_distance(n: int) -> float:
    """Maximum distance any future state can be from present on CP^n.
    Always pi/2, regardless of dimension. The future is BOUNDED.
    """
    return np.pi / 2


def recurrence_time(speed: float) -> float:
    """Time for a state to return to its starting point on CP^n.
    Geodesics on CP^n have period pi (due to phase identification).
    """
    if speed < 1e-15:
        return float('inf')
    return np.pi / speed


def dimension_focusing_comparison(dims: list[int] = None) -> dict:
    """Compare focusing rates across dimensions.
    KEY PREDICTION: larger systems focus faster.

    This is the same effect as the Bures scaling:
    higher dimension -> stronger curvature correction ->
    more constrained future.
    """
    if dims is None:
        dims = [1, 2, 3, 4, 7, 15, 31, 63]  # CP^n for 1-6 qubit systems

    results = {}
    for n in dims:
        t_caustic_holo = conjugate_point(4.0)  # Holomorphic
        t_caustic_real = conjugate_point(1.0)   # Real
        rate = focusing_rate(n)

        results[n] = {
            'cpn_dimension': n,
            'real_dimension': 2 * n,
            'focusing_rate': rate,
            'focal_holo': t_caustic_holo,
            'focal_real': t_caustic_real,
            'ricci': ricci_curvature(n),
            'scalar_curvature': scalar_curvature(n),
        }

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("CP^n Geodesic Prediction Engine")
    print("=" * 60)

    # --- Demonstrate geodesic focusing across dimensions ---
    print("\n--- Focusing Rate vs Dimension ---")
    print(f"{'CP^n':>6} {'Real dim':>8} {'Focusing rate':>14} "
          f"{'Ricci':>8} {'Scalar R':>10}")
    print("-" * 50)

    comparison = dimension_focusing_comparison()
    for n, data in sorted(comparison.items()):
        print(f"CP^{n:>3} {data['real_dimension']:>8} "
              f"{data['focusing_rate']:>14.2f} "
              f"{data['ricci']:>8.1f} {data['scalar_curvature']:>10.1f}")

    # --- Demonstrate geodesic on CP^1 (Bloch sphere) ---
    print("\n--- Geodesic on CP^1 (Bloch sphere) ---")
    psi = normalize(np.array([1.0, 0.0], dtype=complex))  # |0>
    v = normalize(np.array([0.0, 1.0], dtype=complex))     # toward |1>

    print(f"Start: |0>")
    print(f"Direction: toward |1>")
    for t in [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]:
        state = geodesic(psi, v, t)
        d = fubini_study_distance(psi, state)
        print(f"  t={t:.4f}: d_FS={d:.4f}, "
              f"|<0|state>|^2={abs(np.vdot(psi, state))**2:.4f}")

    # --- Demonstrate Jacobi field focusing ---
    print("\n--- Jacobi Field Focusing (K=4, holomorphic) ---")
    print("Nearby geodesics RECONVERGE at t = pi/2:")
    for t in np.linspace(0, np.pi, 9):
        j = jacobi_field(t, curvature=4.0)
        bar = '#' * int(abs(j) * 40)
        print(f"  t={t:.4f}: J={j:>8.4f} |{bar}")

    # --- Conjugate points ---
    print("\n--- Conjugate Points (where geodesics reconverge) ---")
    print(f"  Holomorphic (K=4): t = {conjugate_point(4.0):.4f} = pi/2")
    print(f"  Real (K=1):        t = {conjugate_point(1.0):.4f} = pi")
    print(f"  Flat (K=0):        t = {conjugate_point(0.0)} (never)")

    # --- Prediction demo ---
    print("\n--- Prediction Demo: 3-qubit system (CP^7) ---")
    n = 7
    psi = random_state(n)
    v = random_tangent(psi)

    # Generate "observed" states along a geodesic with small noise
    observed = []
    for i in range(5):
        t = 0.05 * i
        state = geodesic(psi, v, t)
        # Add small noise
        noise = (np.random.randn(n + 1) + 1j * np.random.randn(n + 1)) * 0.001
        state = normalize(state + noise)
        observed.append(state)

    result = convergence_prediction(observed, n_future=10)
    print(f"  Fit quality: {result['fit_quality']:.4f}")
    print(f"  Speed: {result['speed']:.4f} rad/step")
    print(f"  Focusing rate: {result['focusing_rate']:.2f}")
    print(f"  Recurrence time: {recurrence_time(result['speed']):.1f} steps")
    print(f"\n  Future predictions:")
    for i, (state, conf) in enumerate(
            zip(result['predicted_states'], result['confidence'])):
        actual = geodesic(psi, v, 0.05 * (5 + i))
        error = fubini_study_distance(state, actual)
        print(f"    Step +{i+1}: confidence={conf:.3f}, "
              f"actual_error={error:.6f}")

    # --- The key insight ---
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Focusing rate scales with dimension")
    print("=" * 60)
    print(f"\n  1-qubit (CP^1):  focusing = {focusing_rate(1):>6.1f}")
    print(f"  2-qubit (CP^3):  focusing = {focusing_rate(3):>6.1f}")
    print(f"  3-qubit (CP^7):  focusing = {focusing_rate(7):>6.1f}")
    print(f"  4-qubit (CP^15): focusing = {focusing_rate(15):>6.1f}")
    print(f"  5-qubit (CP^31): focusing = {focusing_rate(31):>6.1f}")
    print(f"  6-qubit (CP^63): focusing = {focusing_rate(63):>6.1f}")
    print(f"\n  Ratio 3q/1q: {focusing_rate(7)/focusing_rate(1):.1f}x")
    print(f"  Ratio 6q/1q: {focusing_rate(63)/focusing_rate(1):.1f}x")
    print(f"\n  Same scaling as the Bures advantage on IBM hardware.")
    print(f"  Same curvature. Same K=4. Different manifestation.")
    print(f"\n  The future gets MORE constrained as systems get larger.")
    print(f"  Not less. More.")
