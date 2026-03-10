#!/usr/bin/env python3
"""
HAIQ — Hybrid Assignment via Integrated Quantum computing
==========================================================
Distributed quantum-classical framework for network assignment optimization.

Architecture
------------
1. Network Decomposition : split global problem into geographic cells
2. Local Quantum Solver  : CQF-style Grover oracle per cell (bounded-scale QPU)
3. Classical Communication: measure boundary UEs → feedforward constraints
4. Reconciliation         : combine local solutions into global assignment

Demo scenario
-------------
  6 UEs, 2 APs (a, b),  access limit U_a = U_b = 3
  Power cost matrix P[i, j]  (minimize total power)

  Decomposed into 2 cells:
    QPU_A : UE {0, 1, 2}  (UE 2 = boundary)
    QPU_B : UE {3, 4, 5}  (UE 3 = boundary)

  Classical communication:
    QPU_A measures → count_a (# UEs choosing AP a in cell A)
    This adjusts the access limit for QPU_B:
      AP a can accept at most (3 − count_a) more UEs from cell B
      AP b can accept at most (3 − (3 − count_a)) = count_a more UEs from cell B

Inspired by:
  - CQF  (Jang et al., IEEE Commun. Mag. 2025)   : oracle + diffusion framework
  - QTG  (Wilkening et al., npj Quantum Inf. 2025): tree-based feasible generation
  - DQML (Hwang et al., Quantum Sci. Technol. 2025): CC between QPUs

Author: Yong Hun Jang  |  Korea University
"""

import numpy as np
from itertools import product as iterproduct
from collections import Counter
from dataclasses import dataclass, field

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector


# ╔══════════════════════════════════════════════════════════════╗
# ║  Part 1 — Network Model & Decomposition                    ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class NetworkModel:
    """Global network definition."""
    n_ue: int                          # total UEs
    n_ap: int                          # total APs
    power: np.ndarray                  # P[i,j] = power cost for UE i → AP j
    access_limit: list[int]            # U_a for each AP
    ap_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.ap_names:
            self.ap_names = [f"AP_{chr(97+j)}" for j in range(self.n_ap)]

    def evaluate(self, assignment: list[int]) -> float:
        """Total power cost for a given assignment vector (UE i → AP assignment[i])."""
        return sum(self.power[i, assignment[i]] for i in range(self.n_ue))

    def is_feasible(self, assignment: list[int]) -> bool:
        """Check access limit constraints."""
        counts = Counter(assignment)
        return all(counts.get(j, 0) <= self.access_limit[j]
                   for j in range(self.n_ap))

    def exhaustive_search(self) -> tuple[list[int], float]:
        """Brute-force optimal (for validation only)."""
        best_assign, best_cost = None, float('inf')
        for assign in iterproduct(range(self.n_ap), repeat=self.n_ue):
            assign = list(assign)
            if self.is_feasible(assign):
                c = self.evaluate(assign)
                if c < best_cost:
                    best_cost = c
                    best_assign = assign
        return best_assign, best_cost


@dataclass
class CellProblem:
    """A local subproblem for one geographic cell."""
    cell_id: int
    ue_indices: list[int]              # global UE indices handled by this cell
    n_ap: int
    power_local: np.ndarray            # P_local[local_i, j]
    access_limit_local: list[int]      # adjusted per-AP limits for this cell
    boundary_ue_local: list[int]       # local indices of boundary UEs

    @property
    def n_ue_local(self) -> int:
        return len(self.ue_indices)


def decompose_network(net: NetworkModel,
                      cell_ue_map: dict[int, list[int]]) -> list[CellProblem]:
    """
    Decompose global network into cell subproblems.

    Parameters
    ----------
    net : NetworkModel
    cell_ue_map : {cell_id: [global_ue_indices]}
        Each cell is responsible for a set of UEs.
        UEs appearing in multiple cells are boundary UEs.

    Returns
    -------
    list[CellProblem] with per-cell access limits set to the global limits
    (will be adjusted later via classical communication).
    """
    # 어떤 UE가 여러 셀에 등장하면 boundary
    ue_to_cells = {}
    for cid, ues in cell_ue_map.items():
        for u in ues:
            ue_to_cells.setdefault(u, []).append(cid)

    cells = []
    for cid, ue_list in cell_ue_map.items():
        local_power = net.power[ue_list, :]            # shape (n_local, n_ap)
        boundary_local = [i for i, u in enumerate(ue_list)
                          if len(ue_to_cells[u]) > 1]

        cells.append(CellProblem(
            cell_id=cid,
            ue_indices=ue_list,
            n_ap=net.n_ap,
            power_local=local_power,
            access_limit_local=list(net.access_limit),  # 초기값; CC로 갱신
            boundary_ue_local=boundary_local,
        ))
    return cells


# ╔══════════════════════════════════════════════════════════════╗
# ║  Part 2 — Local Quantum Solver  (CQF-style Grover)         ║
# ╚══════════════════════════════════════════════════════════════╝

def _feasible_patterns(n_ue: int, n_ap: int,
                       access_limits: list[int]) -> list[tuple[int, ...]]:
    """
    Enumerate all feasible assignment patterns for the local problem.
    pattern[i] ∈ {0, ..., n_ap-1} for each local UE i.
    """
    patterns = []
    for assign in iterproduct(range(n_ap), repeat=n_ue):
        counts = Counter(assign)
        if all(counts.get(j, 0) <= access_limits[j] for j in range(n_ap)):
            patterns.append(assign)
    return patterns


def _build_oracle_phase_rotation(qc, assign_reg, cost_reg, anc,
                                 power_matrix, alpha=1.0):
    """
    CQF Phase-rotation oracle: encode continuous power cost into cost qubit phase.
    For minimization: lower power → θ closer to π → higher P(cost=|1⟩).

    동작: 각 UE i에 대해, 각 AP j 조합이 match하면
          cost[i]를 θ = α·π·(1 − P_norm[i,j]) 만큼 회전.
    """
    P = np.asarray(power_matrix, dtype=float)
    n_ue, n_ap = P.shape
    p_min, p_max = float(P.min()), float(P.max())

    if p_max - p_min < 1e-12:
        return

    bits_per_ue = max(1, int(np.ceil(np.log2(n_ap)))) if n_ap > 2 else 1

    for i in range(n_ue):
        for j in range(n_ap):
            # Normalized: 0 (worst) → 1 (best, lowest power)
            p_norm = (P[i, j] - p_min) / (p_max - p_min)
            theta = alpha * np.pi * (1.0 - p_norm)      # 최소 power → θ ≈ π

            if abs(theta) < 1e-12:
                continue

            # Match UE i → AP j: 큐빗 패턴 확인
            if n_ap == 2:
                # 1 qubit per UE: q=0 → AP0, q=1 → AP1
                q = assign_reg[i]
                if j == 0:
                    qc.x(q)
                    qc.cry(theta, q, cost_reg[i])
                    qc.x(q)
                else:
                    qc.cry(theta, q, cost_reg[i])
            else:
                # Multi-qubit encoding per UE
                qs = [assign_reg[i * bits_per_ue + b] for b in range(bits_per_ue)]
                ap_bits = format(j, f'0{bits_per_ue}b')
                flips = [b for b, bit in enumerate(ap_bits) if bit == '0']
                for b in flips:
                    qc.x(qs[b])
                # controlled-Ry on cost qubit
                if len(qs) == 1:
                    qc.cry(theta, qs[0], cost_reg[i])
                else:
                    # multi-controlled Ry via ancilla
                    qc.mcx(qs, anc[0])
                    qc.cry(theta, anc[0], cost_reg[i])
                    qc.mcx(qs, anc[0])
                for b in reversed(flips):
                    qc.x(qs[b])

    qc.barrier()


def _build_inverse_oracle_phase_rotation(qc, assign_reg, cost_reg, anc,
                                         power_matrix, alpha=1.0):
    """Uncomputation: same structure, negative angles."""
    P = np.asarray(power_matrix, dtype=float)
    n_ue, n_ap = P.shape
    p_min, p_max = float(P.min()), float(P.max())
    if p_max - p_min < 1e-12:
        return

    bits_per_ue = max(1, int(np.ceil(np.log2(n_ap)))) if n_ap > 2 else 1

    for i in range(n_ue):
        for j in range(n_ap):
            p_norm = (P[i, j] - p_min) / (p_max - p_min)
            theta = alpha * np.pi * (1.0 - p_norm)
            if abs(theta) < 1e-12:
                continue

            if n_ap == 2:
                q = assign_reg[i]
                if j == 0:
                    qc.x(q)
                    qc.cry(-theta, q, cost_reg[i])
                    qc.x(q)
                else:
                    qc.cry(-theta, q, cost_reg[i])
            else:
                qs = [assign_reg[i * bits_per_ue + b] for b in range(bits_per_ue)]
                ap_bits = format(j, f'0{bits_per_ue}b')
                flips = [b for b, bit in enumerate(ap_bits) if bit == '0']
                for b in flips:
                    qc.x(qs[b])
                if len(qs) == 1:
                    qc.cry(-theta, qs[0], cost_reg[i])
                else:
                    qc.mcx(qs, anc[0])
                    qc.cry(-theta, anc[0], cost_reg[i])
                    qc.mcx(qs, anc[0])
                for b in reversed(flips):
                    qc.x(qs[b])

    qc.barrier()


def _build_constraint_oracle(qc, assign_reg, cost_reg, superflag,
                             feasible_patterns, n_ue, n_ap):
    """
    Hard-constraint oracle: phase-flip feasible states where all cost qubits = |1⟩.

    For each feasible pattern:
      1) Toggle ancilla if assign matches pattern
      2) MCX(all cost qubits + ancilla → superflag) for phase kickback
      3) Uncompute ancilla

    feasible_patterns에 포함된 assignment만 마킹됨.
    """
    bits_per_ue = 1 if n_ap == 2 else max(1, int(np.ceil(np.log2(n_ap))))
    n_assign = n_ue * bits_per_ue

    # cost 큐빗 전체 + 패턴 매치용 ancilla (cost[-1]을 임시 재활용하는 대신,
    # 별도 ancilla 필요 — 여기서는 superflag와 cost의 MCX로 처리)

    for pattern in feasible_patterns:
        # Step 1: X-mask to match pattern
        masked = []
        for i, ap_idx in enumerate(pattern):
            if n_ap == 2:
                q = assign_reg[i]
                if ap_idx == 0:
                    qc.x(q)
                    masked.append(q)
            else:
                for b in range(bits_per_ue):
                    q = assign_reg[i * bits_per_ue + b]
                    bit_val = (ap_idx >> (bits_per_ue - 1 - b)) & 1
                    if bit_val == 0:
                        qc.x(q)
                        masked.append(q)

        # Step 2: MCX — all assign + all cost → superflag (phase kickback)
        controls = list(assign_reg[:n_assign]) + list(cost_reg[:n_ue])
        qc.mcx(controls, superflag[0])

        # Step 3: undo X-mask
        for q in reversed(masked):
            qc.x(q)

    qc.barrier()


def _build_diffusion(qc, assign_reg, n_qubits):
    """Standard Grover diffusion operator about |ψ₀⟩ = H^n|0⟩."""
    qubits = assign_reg[:n_qubits]
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(list(qubits[:-1]), qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    qc.barrier()


def build_local_solver(cell: CellProblem,
                       alpha: float = 1.0,
                       iterations: int = 1) -> QuantumCircuit:
    """
    Build a complete CQF Grover circuit for one cell's local assignment problem.

    Returns a QuantumCircuit ready for simulation or execution.
    """
    n = cell.n_ue_local
    n_ap = cell.n_ap

    bits_per_ue = 1 if n_ap == 2 else max(1, int(np.ceil(np.log2(n_ap))))
    n_assign = n * bits_per_ue

    # --- Registers ---
    assign_reg = QuantumRegister(n_assign, "assign")
    cost_reg   = QuantumRegister(n, "cost")
    anc        = QuantumRegister(1, "anc")
    superflag  = QuantumRegister(1, "sf")
    c_reg      = ClassicalRegister(n_assign, "c")

    qc = QuantumCircuit(assign_reg, cost_reg, anc, superflag, c_reg)

    # --- Preparation ---
    qc.h(assign_reg)
    qc.x(superflag)
    qc.h(superflag)       # |−⟩ for phase kickback
    qc.barrier()

    # --- Feasible patterns ---
    patterns = _feasible_patterns(n, n_ap, cell.access_limit_local)

    # --- Grover iterations ---
    for _ in range(iterations):
        # Oracle: phase rotation + constraint marking + uncompute
        _build_oracle_phase_rotation(
            qc, assign_reg, cost_reg, anc,
            cell.power_local, alpha)

        _build_constraint_oracle(
            qc, assign_reg, cost_reg, superflag,
            patterns, n, n_ap)

        _build_inverse_oracle_phase_rotation(
            qc, assign_reg, cost_reg, anc,
            cell.power_local, alpha)

        # Diffusion
        _build_diffusion(qc, assign_reg, n_assign)

    # --- Measurement ---
    qc.measure(assign_reg, c_reg)

    return qc


# ╔══════════════════════════════════════════════════════════════╗
# ║  Part 3 — Classical Communication & Reconciliation          ║
# ╚══════════════════════════════════════════════════════════════╝

def decode_local_result(bitstring: str, n_ue: int, n_ap: int) -> list[int]:
    """
    Qiskit 측정 bitstring → AP assignment list.
    Qiskit bitstring: 오른쪽이 q[0] (LSB).
    """
    bits = bitstring[::-1]  # q[0]부터 순서대로
    if n_ap == 2:
        return [int(bits[i]) for i in range(n_ue)]
    else:
        bpu = max(1, int(np.ceil(np.log2(n_ap))))
        assign = []
        for i in range(n_ue):
            val = int(bits[i*bpu:(i+1)*bpu], 2)
            assign.append(min(val, n_ap - 1))
        return assign


def classical_communication(cell_a_result: list[int],
                            cell_a: CellProblem,
                            cell_b: CellProblem,
                            global_limits: list[int]) -> list[int]:
    """
    CC protocol (DQML-inspired):
      1) Count how many UEs in cell A chose each AP
      2) Compute remaining capacity for cell B
      3) Return adjusted access limits for cell B

    이것이 mid-circuit measurement → feedforward 의 classical simulation.
    실제 양자 하드웨어에서는 QPU_A의 mid-circuit measurement 결과를
    classical processor를 거쳐 QPU_B의 constraint parameter로 전달.
    """
    # Cell A의 AP별 사용량 집계
    counts_a = Counter(cell_a_result)

    # Cell B의 access limit 갱신: 글로벌 한도 − cell A 사용량
    adjusted_limits = []
    for j in range(cell_b.n_ap):
        remaining = global_limits[j] - counts_a.get(j, 0)
        adjusted_limits.append(max(0, remaining))

    return adjusted_limits


def reconcile_global(cell_results: dict[int, list[int]],
                     cell_problems: list[CellProblem],
                     net: NetworkModel) -> list[int]:
    """
    Combine local cell results into a global assignment.
    Boundary UE 충돌 해결: 더 낮은 power cost를 주는 쪽 선택.
    """
    global_assign = [None] * net.n_ue

    # 각 셀 결과 수집
    ue_candidates = {}   # global_ue_idx → [(ap, power_cost, cell_id), ...]
    for cell in cell_problems:
        result = cell_results[cell.cell_id]
        for local_i, global_i in enumerate(cell.ue_indices):
            ap = result[local_i]
            cost = net.power[global_i, ap]
            ue_candidates.setdefault(global_i, []).append((ap, cost, cell.cell_id))

    # 비경계 UE: 직접 할당, 경계 UE: 최소 cost 선택 (majority vote 변형)
    for ue_idx in range(net.n_ue):
        candidates = ue_candidates.get(ue_idx, [])
        if len(candidates) == 1:
            global_assign[ue_idx] = candidates[0][0]
        elif len(candidates) > 1:
            # 최소 power cost 선택
            best = min(candidates, key=lambda x: x[1])
            global_assign[ue_idx] = best[0]
        else:
            global_assign[ue_idx] = 0   # fallback

    return global_assign


# ╔══════════════════════════════════════════════════════════════╗
# ║  Part 4 — Centralized Solver (baseline comparison)          ║
# ╚══════════════════════════════════════════════════════════════╝

def build_centralized_solver(net: NetworkModel,
                             alpha: float = 1.0,
                             iterations: int = 1) -> QuantumCircuit:
    """
    기존 CQF 방식: 모든 UE를 하나의 QPU에서 처리.
    """
    cell_all = CellProblem(
        cell_id=0,
        ue_indices=list(range(net.n_ue)),
        n_ap=net.n_ap,
        power_local=net.power,
        access_limit_local=list(net.access_limit),
        boundary_ue_local=[],
    )
    return build_local_solver(cell_all, alpha, iterations)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Part 5 — Hybrid Pipeline                                   ║
# ╚══════════════════════════════════════════════════════════════╝

class HybridPipeline:
    """
    Full hybrid quantum-classical pipeline.

    Usage:
        pipe = HybridPipeline(network, cell_ue_map)
        result = pipe.run(shots=4096, iterations=1)
    """

    def __init__(self, net: NetworkModel, cell_ue_map: dict[int, list[int]]):
        self.net = net
        self.cell_ue_map = cell_ue_map
        self.cells = decompose_network(net, cell_ue_map)
        self.backend = Aer.get_backend('aer_simulator')

    def _run_cell(self, cell: CellProblem,
                  shots: int, alpha: float, iterations: int) -> dict:
        """Run one cell's local solver and return measurement counts."""
        qc = build_local_solver(cell, alpha, iterations)
        transpiled = transpile(qc, self.backend)
        result = self.backend.run(transpiled, shots=shots).result()
        return result.get_counts()

    def _most_probable(self, counts: dict, n_ue: int, n_ap: int) -> list[int]:
        """가장 높은 shot count를 가진 assignment 추출."""
        best_bits = max(counts, key=counts.get)
        return decode_local_result(best_bits, n_ue, n_ap)

    def _run_single_pass(self, shots, alpha, iterations, verbose=False):
        """Execute one forward pass through all cells with CC."""
        cell_results = {}
        qubit_counts = {}
        prev_counts_per_ap = None

        for idx, cell in enumerate(self.cells):
            # --- CC: 이전 셀 결과로 access limit 조정 ---
            if prev_counts_per_ap is not None:
                adjusted = []
                for j in range(cell.n_ap):
                    remaining = self.net.access_limit[j] - prev_counts_per_ap.get(j, 0)
                    adjusted.append(max(0, remaining))
                cell.access_limit_local = adjusted
                if verbose:
                    print(f"    [CC] Cell {cell.cell_id} limits: {adjusted}")

            if verbose:
                n_q = cell.n_ue_local if cell.n_ap == 2 else \
                      cell.n_ue_local * int(np.ceil(np.log2(cell.n_ap)))
                total_q = n_q + cell.n_ue_local + 1 + 1
                print(f"    [QPU {cell.cell_id}] UEs={cell.ue_indices}, qubits={total_q}")

            counts = self._run_cell(cell, shots, alpha, iterations)

            local_assign = self._most_probable(counts, cell.n_ue_local, cell.n_ap)
            cell_results[cell.cell_id] = local_assign
            prev_counts_per_ap = Counter(local_assign)

            if verbose:
                local_cost = sum(cell.power_local[i, local_assign[i]]
                                 for i in range(cell.n_ue_local))
                print(f"      → {dict(zip(cell.ue_indices, local_assign))} "
                      f"cost={local_cost:.2f}")

            qc_tmp = build_local_solver(cell, alpha, iterations)
            qubit_counts[cell.cell_id] = qc_tmp.num_qubits

        global_assign = reconcile_global(cell_results, self.cells, self.net)
        total_power = self.net.evaluate(global_assign)
        feasible = self.net.is_feasible(global_assign)

        return global_assign, total_power, feasible, cell_results, qubit_counts

    def run(self, shots: int = 4096, alpha: float = 1.0,
            iterations: int = 1, cc_rounds: int = 1,
            verbose: bool = True) -> dict:
        """
        Execute the hybrid pipeline with iterative CC refinement.

        Parameters
        ----------
        cc_rounds : int
            Number of CC refinement rounds.  Round 1 = forward pass only.
            Round 2+ = re-run each cell with updated boundary awareness.
            Multiple rounds 수행 시, 각 round에서 best-so-far 해를
            다음 round의 경계 조건으로 피드백.

        Returns
        -------
        dict with keys: 'global_assignment', 'total_power', 'feasible',
                        'cell_results', 'qubit_counts', 'history'
        """
        if verbose:
            print("=" * 60)
            print(f"  HAIQ Hybrid Pipeline  ({cc_rounds} CC round(s))")
            print("=" * 60)

        best_assign = None
        best_cost = float('inf')
        best_feasible = False
        best_cell_results = {}
        best_qubit_counts = {}
        history = []

        # 셀 decomposition 초기화 (access limit을 매 라운드 리셋하기 위해 보관)
        original_limits = [list(c.access_limit_local) for c in self.cells]

        for rnd in range(cc_rounds):
            if verbose:
                print(f"\n  ── Round {rnd + 1}/{cc_rounds} ──")

            # 매 라운드 시작 전 limits 리셋
            for c, lim in zip(self.cells, original_limits):
                c.access_limit_local = list(lim)

            g_assign, g_cost, g_feas, c_res, q_counts = \
                self._run_single_pass(shots, alpha, iterations,
                                      verbose=verbose)

            history.append({
                'round': rnd + 1,
                'assignment': g_assign,
                'cost': g_cost,
                'feasible': g_feas,
            })

            if verbose:
                ap_names = self.net.ap_names
                readable = [f"UE{i}→{ap_names[a]}" for i, a in enumerate(g_assign)]
                print(f"    Global: {readable}  cost={g_cost:.2f}  feas={g_feas}")

            # Best-so-far 갱신
            if g_feas and g_cost < best_cost:
                best_assign = g_assign
                best_cost = g_cost
                best_feasible = g_feas
                best_cell_results = c_res
                best_qubit_counts = q_counts

        if verbose:
            print(f"\n{'=' * 60}")
            ap_names = self.net.ap_names
            readable = [f"UE{i}→{ap_names[a]}" for i, a in enumerate(best_assign)]
            print(f"  Best Assignment : {best_assign}")
            print(f"  Readable        : {readable}")
            print(f"  Total Power     : {best_cost:.4f}")
            print(f"  Feasible        : {best_feasible}")
            print(f"  Max qubits/QPU  : {max(best_qubit_counts.values())}")
            print(f"{'=' * 60}")

        return {
            'global_assignment': best_assign,
            'total_power': best_cost,
            'feasible': best_feasible,
            'cell_results': best_cell_results,
            'qubit_counts': best_qubit_counts,
            'history': history,
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  Part 6 — Demo & Comparison                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def run_demo():
    """
    6UE / 2AP 시나리오에서 centralized vs hybrid 비교.
    """
    # --- Network definition ---
    #        AP_a   AP_b
    # UE0    10.85  19.30    (close to AP_a)
    # UE1     8.45  17.45    (close to AP_a)
    # UE2     9.15   8.20    (boundary — similar cost)
    # UE3     9.25  10.40    (boundary — similar cost)
    # UE4    18.50   8.90    (close to AP_b)
    # UE5    16.20   9.60    (close to AP_b)

    power = np.array([
        [10.85, 19.30],   # UE0 — clearly prefers AP_a
        [ 8.45, 17.45],   # UE1 — clearly prefers AP_a
        [ 9.15,  8.20],   # UE2 — boundary (slight preference for AP_b)
        [ 9.25, 10.40],   # UE3 — boundary (slight preference for AP_a)
        [18.50,  8.90],   # UE4 — clearly prefers AP_b
        [16.20,  9.60],   # UE5 — clearly prefers AP_b
    ])

    net = NetworkModel(
        n_ue=6, n_ap=2,
        power=power,
        access_limit=[3, 3],    # 각 AP 최대 3 UE
        ap_names=["AP_a", "AP_b"],
    )

    # ========== 1. Classical exhaustive search ==========
    print("\n" + "▓" * 60)
    print("  1. CLASSICAL EXHAUSTIVE SEARCH (ground truth)")
    print("▓" * 60)

    opt_assign, opt_cost = net.exhaustive_search()
    readable = [f"UE{i}→{net.ap_names[a]}" for i, a in enumerate(opt_assign)]
    print(f"  Optimal  : {opt_assign}")
    print(f"  Readable : {readable}")
    print(f"  Cost     : {opt_cost:.4f}")
    print(f"  Feasible : {net.is_feasible(opt_assign)}")

    # ========== 2. Centralized quantum (all 6 UEs in one QPU) ==========
    print("\n" + "▓" * 60)
    print("  2. CENTRALIZED CQF  (6 UEs → 1 QPU)")
    print("▓" * 60)

    qc_central = build_centralized_solver(net, alpha=1.0, iterations=2)
    print(f"  Total qubits: {qc_central.num_qubits}")

    backend = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc_central, backend)
    result = backend.run(transpiled, shots=8192).result()
    counts = result.get_counts()

    best_bits = max(counts, key=counts.get)
    central_assign = decode_local_result(best_bits, net.n_ue, net.n_ap)
    central_cost = net.evaluate(central_assign)
    central_feas = net.is_feasible(central_assign)

    readable = [f"UE{i}→{net.ap_names[a]}" for i, a in enumerate(central_assign)]
    print(f"  Assignment: {central_assign}")
    print(f"  Readable  : {readable}")
    print(f"  Cost      : {central_cost:.4f}")
    print(f"  Feasible  : {central_feas}")

    # Top-5
    print(f"\n  Top-5 measured states:")
    for bits, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        a = decode_local_result(bits, net.n_ue, net.n_ap)
        c = net.evaluate(a)
        f = net.is_feasible(a)
        print(f"    {bits} count={cnt:5d} assign={a} cost={c:.2f} feas={f}")

    # ========== 3. Hybrid quantum-classical (2 QPUs with CC) ==========
    print("\n" + "▓" * 60)
    print("  3. HYBRID HAIQ  (6 UEs → 2 QPUs with CC)")
    print("▓" * 60)

    # Cell decomposition:
    #   Cell 0 (QPU_A): UE 0, 1, 2   (2 local + 1 boundary)
    #   Cell 1 (QPU_B): UE 3, 4, 5   (1 boundary + 2 local)
    cell_map = {
        0: [0, 1, 2],   # QPU_A
        1: [3, 4, 5],   # QPU_B
    }

    pipeline = HybridPipeline(net, cell_map)
    hybrid_result = pipeline.run(shots=8192, alpha=1.0, iterations=2,
                                 cc_rounds=10)

    # ========== 4. Summary ==========
    print("\n" + "▓" * 60)
    print("  COMPARISON SUMMARY")
    print("▓" * 60)

    gap_central = abs(central_cost - opt_cost)
    gap_hybrid  = abs(hybrid_result['total_power'] - opt_cost)

    print(f"  {'Method':<25} {'Cost':>10} {'Gap':>10} {'Feas':>6} {'Qubits':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*6} {'-'*8}")
    print(f"  {'Classical optimal':<25} {opt_cost:10.4f} {'—':>10} "
          f"{'✓':>6} {'—':>8}")
    print(f"  {'Centralized CQF':<25} {central_cost:10.4f} {gap_central:10.4f} "
          f"{'✓' if central_feas else '✗':>6} {qc_central.num_qubits:>8}")
    max_q = max(hybrid_result['qubit_counts'].values())
    print(f"  {'Hybrid HAIQ (2 QPU)':<25} {hybrid_result['total_power']:10.4f} "
          f"{gap_hybrid:10.4f} "
          f"{'✓' if hybrid_result['feasible'] else '✗':>6} {max_q:>8}")

    print()

    # ========== 5. Scalability projection ==========
    print("▓" * 60)
    print("  SCALABILITY PROJECTION")
    print("▓" * 60)

    print(f"\n  {'N_UE':>6} {'Centralized':>14} {'Hybrid(k=3)':>14} {'Reduction':>10}")
    print(f"  {'-'*6} {'-'*14} {'-'*14} {'-'*10}")

    for n_ue in [6, 12, 24, 48, 96, 192]:
        # Centralized: n_ue assign + n_ue cost + 1 anc + 1 sf  (2-AP case)
        q_central = n_ue + n_ue + 1 + 1
        # Hybrid: k cells each with n_ue/k UEs
        k = max(2, n_ue // 3)         # ~3 UEs per cell
        n_local = max(3, n_ue // k)
        q_hybrid = n_local + n_local + 1 + 1
        reduction = (1 - q_hybrid / q_central) * 100
        print(f"  {n_ue:>6} {q_central:>14} {q_hybrid:>14} {reduction:>9.1f}%")

    print()
    return opt_assign, opt_cost


# ╔══════════════════════════════════════════════════════════════╗
# ║  Entry point                                                ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    run_demo()
