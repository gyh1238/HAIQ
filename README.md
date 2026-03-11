# HAIQ — Hybrid Assignment via Integrated Quantum Computing

Distributed quantum-classical framework for wireless network assignment optimization.
Extends the CQF (Comprehensive Quantum Framework) paper by incorporating distributed quantum computing with classical coordination.

## Architecture

```
Global Network
     |
     v
 ┌──────────┐     ┌──────────┐     ┌──────────┐
 │  Zone 0   │     │  Zone 1   │     │  Zone 2   │
 │  (QPU 0)  │     │  (QPU 1)  │     │  (QPU 2)  │
 │ UE0,1,2*  │     │ UE2*,3,4* │     │ UE4*,5   │
 └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │   * = boundary  │
                v                 v
        ┌──────────────┐  ┌──────────────┐
        │   Collect     │  │   Resolve    │
        │  Candidates   │→│  Boundaries  │
        └──────────────┘  └──────┬───────┘
                                 │
                                 v
                        Global Assignment
```

Boundary UEs are duplicated across zones.
Each QPU solves its local selection problem independently (parallel execution).
Post-measurement classical reconciliation resolves boundary conflicts.

## Project Structure

```
HAIQ/
├── CQF/                        # Original CQF paper implementation (Qiskit 1.1)
│   ├── inter_assignment_qiskit1.1.ipynb   # 4UE/2AP inter-cell (Grover)
│   ├── intra_assignment_qiskit1.1.ipynb   # Intra-cell RB assignment
│   ├── hardcoding_to_c.py                 # 4UE×3BS extended scenario
│   └── test_*.ipynb                       # Constraint & weight tests
│
├── QTG/                        # QTG-style feasible state generation (WIP)
│   ├── QFT_ADD_SUB.ipynb                  # QFT Draper adder utilities
│   └── CAP_*.ipynb                        # Progressive QTG development
│
├── hybrid/                     # Distributed quantum network assignment
│   ├── distributed_assignment.py          # DQNA main module (active)
│   ├── dqna_test.ipynb                    # Test & visualization notebook
│   ├── hybrid_framework.py               # CC-based approach (legacy)
│   └── hybrid_assignment_example.py       # Early prototype (legacy)
│
├── CLAUDE.md                   # Project conventions & instructions
└── README.md
```

## Hybrid Module: DQNA

The primary implementation in `hybrid/distributed_assignment.py`.

### Scenario
- 6 UE × 3 AP, AP capacity = 2 each
- 3 zones with boundary UE overlap: UE2 (Zone 0/1), UE4 (Zone 1/2)

### Quantum Circuit Design: Biased Grover

```
|0⟩ ── Ry(θ₀) ──┤            ├── Ry(-θ₀) ── X ── MCZ ── X ── Ry(θ₀) ── ...
|0⟩ ── Ry(θ₁) ──┤  Constraint├── Ry(-θ₁) ── X ── MCZ ── X ── Ry(θ₁) ── ...
|0⟩ ── Ry(θ₂) ──┤   Oracle   ├── Ry(-θ₂) ── X ── MCZ ── X ── Ry(θ₂) ── ...
|0⟩ ── flag  ───┤            ├───────────────────────────────────────── ...
|0⟩ ── X ── H ──┤  (|−⟩ sf)  ├───────────────────────────────────────── ...
                 └────────────┘
      Biased Init     Oracle          Biased Diffusion
```

- **Quality encoding**: Ry-biased initial state (higher quality → more P(|1⟩))
- **Constraint oracle**: Phase-flip feasible states (min_assign ≤ hw ≤ capacity)
- **Biased diffusion**: Reflection about quality-weighted initial state (A†·MCZ·A)
- **Separation principle**: Quality in initial state, constraints in oracle — no destructive interference

### Pipeline

1. **Partition**: Network → zones with boundary UE detection
2. **Zone Solve**: Each zone runs biased Grover circuit independently
3. **Collect**: Gather per-UE candidates from all zone measurements
4. **Resolve**: Quality-based greedy reconciliation for boundary conflicts
5. **Multi-round**: Repeat with top-K combination search for robustness

### Performance (6UE × 3AP)

| Method | Score | Ratio | Qubits/Zone |
|--------|-------|-------|-------------|
| Classical optimal | 46.0 | 100% | — |
| DQNA (single round) | ~30 | ~66% | 5 |
| DQNA (multi-round) | 46.0 | 100% | 5 |

## Tech Stack

- Python 3.11+
- Qiskit 1.1+ with qiskit_aer
- Statevector (exact) and AerSimulator (shot-based)

## Getting Started

```bash
pip install qiskit==1.1 qiskit_aer pylatexenc

# Run DQNA demo
python hybrid/distributed_assignment.py

# Or use the notebook
jupyter notebook hybrid/dqna_test.ipynb
```

## Research References

- **CQF**: Jang et al., IEEE Commun. Mag. 2025 — Oracle + diffusion framework
- **QTG**: Wilkening et al., npj Quantum Inf. 2025 — Tree-based feasible generation
- **DQML**: Hwang et al., Quantum Sci. Technol. 2025 — CC between QPUs
