# Geodesic Focusing on CP^n and the Quantum Computing Scaling Ceiling

**Paper:** [paper_draft.tex](paper_draft.tex)
**Author:** Nicholas Muir / Roboticc Pty Ltd
**License:** CC BY-NC 4.0 (free for research; commercial use requires license)
**Companion paper:** [cpn-unification](https://github.com/Roboticc-Apps/cpn-unification)

---

## What This Is

Density matrices live on curved manifolds (CP^n). Every quantum computer averages them using flat (Euclidean) geometry. This paper shows:

1. The Raychaudhuri focusing equation on CP^n proves the Euclidean error grows exponentially: focusing rate = 2^(q+1) for q qubits
2. Hardware validation on IBM ibm_fez confirms the scaling: +0.003% (1-qubit), +0.415% (2-qubit), +1.332% (3-qubit)
3. The fix is a post-processing change: replace Euclidean averaging with Bures (Riemannian) mean

## Quick Start

```bash
# Clone
git clone https://github.com/Roboticc-Apps/cpn-prediction.git
cd cpn-prediction

# Install dependencies
pip install numpy scipy matplotlib

# Run validation (61 mathematical checks)
python validate.py

# On Windows, if you get encoding errors:
set PYTHONIOENCODING=utf-8
python validate.py
```

## Hardware Validation

To cross-validate against the real IBM hardware data, you also need the companion repo:

```bash
# Clone the companion repo alongside this one
cd ..
git clone https://github.com/Roboticc-Apps/cpn-unification.git
cd cpn-prediction

# Run hardware validation
python validate_hardware.py
```

## Files

- `paper_draft.tex` -- The paper
- `cpn_geodesic.py` -- Raychaudhuri equation, geodesics, focusing on CP^n
- `calibration.py` -- Bures mean algorithm, fidelity computation
- `scaling_test.py` -- Monte Carlo simulation of Bures vs Euclidean scaling
- `raychaudhuri.py` -- Visualization of geodesic focusing
- `validate.py` -- 61 mathematical validation tests
- `validate_hardware.py` -- Cross-validation against IBM hardware data
- `LICENSE` -- CC BY-NC 4.0

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib

## Citation

```
N. Muir, "Geodesic focusing on CP^n and the quantum computing scaling ceiling,"
(2026). Companion to Australian Patent Application 2026901876.
```
