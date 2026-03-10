# HAIQ — Hybrid Assignment via Integrated Quantum computing

## Project Overview
Quantum-classical hybrid framework for wireless network assignment optimization.
Extends the CQF (Comprehensive Quantum Framework) paper published in IEEE Communications Magazine
by incorporating distributed quantum computing with classical communication.

## Research Goal
Decompose large-scale network assignment problems into localized subproblems,
solve each with bounded-scale quantum circuits, and reconcile outputs via classical coordination.
Key inspirations: QTG (Quantum Tree Generator) for feasible state generation,
DQML (Distributed QML via Classical Communication) for inter-QPU coordination.

## Tech Stack
- **Language**: Python 3.11+
- **Quantum SDK**: Qiskit 1.1+ with qiskit_aer
- **Simulation**: Statevector (exact) and AerSimulator (shot-based)
- **Notebooks**: Jupyter for prototyping, .py scripts for production runs

## Key Directories

### `CQF/` — Original CQF paper implementation (Qiskit 1.1)
- `inter_assignment_qiskit1.1.ipynb`: 4UE/2AP inter-cell assignment with Grover's amplitude amplification.
  Core pipeline: prepare_state_register → oracle_objective_phase → oracle_access_limit → apply_superflag → apply_diffusion → measurement.
- `intra_assignment_qiskit1.1.ipynb`: Intra-cell RB assignment (bipartite matching).
- `hardcoding_to_c.py`: Extended 4UE × 3BS scenario. 2 qubits/UE encoding, 36 feasible patterns (2,1,1 distribution), weight maximization via phase rotation.
- `test_constraint.ipynb`, `test_weight.ipynb`: Constraint oracle and weight encoding tests.

### `QTG/` — QTG-style feasible state generation (work in progress)
- `QFT_ADD_SUB.ipynb`: QFT-based Draper adder/subtractor utilities (shared arithmetic primitives).
- `CAP_without_cond.ipynb` → `CAP_with_cond.ipynb` → `CAP_with_Profit.ipynb` → `CAP_with_Profit_phase.ipynb` → `CAP_realnumber.ipynb`:
  Progressive development of QTG. Each adds capability:
  - capacity checking only → capacity with condition flags → profit register (integer) → profit as phase accumulator → real-number weights.
- Core function: `build_qtg_with_profit(weights, profits, capacity, theta)` — QTG Box 1 implementation.
  For each item m: (a) IntegerComparator checks cap ≥ w_m, (b) conditional Ry creates path superposition, (c) controlled sub/add updates cap/profit registers.

## Quantum Circuit Conventions
- Qubit ordering: Qiskit convention (q[0] = LSB in measurement bitstring).
- QFT arithmetic: MSB → LSB register ordering internally; flip to LSB-first for IntegerComparator.
- Phase rotation for objectives: θ = α·π·w_norm maps continuous utility to qubit phase. Higher utility → phase closer to π → higher P(|1⟩).
- Constraint enforcement: hard constraints via oracle phase-flip (not penalty-based QUBO).
- Amplitude amplification: oracle + diffusion iterations; superflag qubit in |−⟩ for phase kickback.

## Common Patterns in This Codebase

### State preparation
```python
qc.h(assign)          # uniform superposition over all assignments
qc.x(superflag)
qc.h(superflag)       # |−⟩ for phase kickback
```

### QFT-based controlled arithmetic
```python
def controlled_sub_classical_on(circ, control, qubits, a):
    qft_on(circ, qubits)
    for j, qj in enumerate(qubits):
        theta = -2 * np.pi * a / (2 ** (j + 1))
        circ.cp(theta, control, qj)
    iqft_on(circ, qubits)
```

### Statevector debugging
```python
sv = Statevector.from_instruction(qc)
print_full_statevector_clean(sv, threshold=1e-6, forward=True)
```

## Development Commands
```bash
# Install dependencies
pip install qiskit==1.1 qiskit_aer pylatexenc

# Run a notebook as script
jupyter nbconvert --to script notebook.ipynb --stdout | python3

# Quick statevector test
python3 -c "from qiskit.quantum_info import Statevector; ..."
```

## Code Style
- Korean comments are common (bilingual codebase for Korean/English paper writing).
- Function names follow the CQF paper terminology: oracle, diffusion, prepare, amplify.
- Use `np.ndarray` for cost/weight matrices with shape (N_UE, M_AP) or (N_UE, N_RB).
- Always include `qc.barrier()` between circuit modules for visual clarity.
- Register naming: `assign` (state qubits), `cost`/`cap` (capacity/cost), `flag`/`superflag` (constraint marking), `anc_cost`/`wcmp` (ancilla).

## Next Steps (Hybrid Extension)
1. Network decomposition module: partition global network into local subproblems.
2. Local solver: QTG-style feasible state generation adapted for network assignment.
3. Classical communication: mid-circuit measurement + feedforward for inter-cell coordination (inspired by DQML CC scheme).
4. Reconciliation: resolve boundary UE/RB conflicts across subproblems.
5. Benchmark: compare hybrid vs centralized CQF on scalability metrics.

## Important Notes
- Do NOT use QUBO/penalty-based constraint relaxation. This project enforces hard constraints via oracle design.
- Phase rotation encodes continuous objectives — do not quantize to discrete levels.
- Uncomputation is critical: always mirror oracle operations to disentangle auxiliary qubits after marking.
- The 작성자 (author) prefers mathematical rigor over superficial approximations.
