"""
Hybrid Quantum-Classical Network Assignment Example
====================================================
향후 방향 5단계를 간단한 예제로 구현:
  1. Network Decomposition: 글로벌 네트워크를 로컬 서브문제로 분할
  2. Local Solver: 각 서브문제를 CQF 스타일 양자 회로(Grover)로 해결
  3. Classical Communication: 경계 UE 할당 정보를 고전 통신으로 교환
  4. Reconciliation: 경계 UE/AP 충돌 해소
  5. Benchmark: 중앙집중 CQF vs 하이브리드 비교

시나리오: 6 UE, 3 AP (Cell A: AP0, Cell B: AP1, Cell C: AP2)
  - Cell A 커버리지: UE0, UE1, UE2 (UE2는 경계 UE)
  - Cell B 커버리지: UE2, UE3, UE4 (UE2, UE4는 경계 UE)
  - Cell C 커버리지: UE4, UE5
  - 각 AP 최대 수용: 2 UE
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from itertools import product


# ============================================================
# 글로벌 네트워크 정의
# ============================================================

N_UE = 6
N_AP = 3
AP_CAPACITY = 2  # 각 AP 최대 2 UE 수용

# 링크 품질 행렬 (UE × AP), -1 = 연결 불가
# 높을수록 좋은 채널 품질
LINK_QUALITY = np.array([
    [ 9.0, -1.0, -1.0],  # UE0: Cell A만
    [ 7.5, -1.0, -1.0],  # UE1: Cell A만
    [ 4.0,  6.0, -1.0],  # UE2: Cell A/B 경계
    [-1.0,  8.0, -1.0],  # UE3: Cell B만
    [-1.0,  5.0,  7.0],  # UE4: Cell B/C 경계
    [-1.0, -1.0,  8.5],  # UE5: Cell C만
])

# 서브문제 정의: 각 셀이 담당하는 UE 인덱스
CELL_A_UES = [0, 1, 2]   # UE0, UE1, UE2
CELL_B_UES = [2, 3, 4]   # UE2, UE3, UE4
CELL_C_UES = [4, 5]      # UE4, UE5

BOUNDARY_UES = {2, 4}  # 경계 UE


# ============================================================
# Step 1: Network Decomposition (네트워크 분할)
# ============================================================

def decompose_network():
    """글로벌 문제를 셀 단위 서브문제로 분할."""
    subproblems = []

    cells = [
        {"name": "Cell_A", "ues": CELL_A_UES, "ap": 0},
        {"name": "Cell_B", "ues": CELL_B_UES, "ap": 1},
        {"name": "Cell_C", "ues": CELL_C_UES, "ap": 2},
    ]

    for cell in cells:
        ue_indices = cell["ues"]
        ap_idx = cell["ap"]
        # 로컬 링크 품질 추출 (해당 AP에 대한 UE들의 채널 품질)
        local_qualities = []
        for ue in ue_indices:
            q = LINK_QUALITY[ue, ap_idx]
            local_qualities.append(q if q > 0 else 0.0)

        subproblems.append({
            "name": cell["name"],
            "ue_indices": ue_indices,
            "ap_index": ap_idx,
            "qualities": np.array(local_qualities),
            "capacity": AP_CAPACITY,
        })

    return subproblems


# ============================================================
# Step 2: Local Solver — CQF 스타일 Grover 양자 회로
# ============================================================

def build_local_grover(subproblem, iterations=1):
    """
    서브문제에 대한 로컬 Grover 회로 구성.

    각 UE에 1큐비트 (0=해당 AP에 할당 안함, 1=할당).
    Oracle: 용량 제약(AP에 최대 capacity개 UE) + 목적함수(채널 품질) 위상 인코딩.
    """
    n_ue = len(subproblem["ue_indices"])
    qualities = subproblem["qualities"]
    capacity = subproblem["capacity"]

    assign = QuantumRegister(n_ue, name="assign")
    flag = QuantumRegister(1, name="flag")
    cl = ClassicalRegister(n_ue, name="c")
    qc = QuantumCircuit(assign, flag, cl)

    # --- Preparation: 균일 중첩 ---
    qc.h(assign)
    qc.x(flag)
    qc.h(flag)  # |−⟩ for phase kickback
    qc.barrier()

    for _ in range(iterations):
        # --- Oracle ---
        _apply_local_oracle(qc, assign, flag, qualities, capacity)
        qc.barrier()

        # --- Diffusion ---
        qc.h(assign)
        qc.x(assign)
        qc.h(assign[-1])
        if n_ue > 1:
            qc.mcx(assign[:-1], assign[-1])
        qc.h(assign[-1])
        qc.x(assign)
        qc.h(assign)
        qc.barrier()

    qc.measure(assign, cl)
    return qc


def _apply_local_oracle(qc, assign, flag, qualities, capacity):
    """
    Oracle: 실현 가능하고 높은 품질인 상태를 선호하도록 위상 마킹.

    1) 용량 제약 위반(할당 UE > capacity) 상태에 위상 반전 (hard constraint).
    2) 목적함수: 채널 품질을 위상 회전으로 인코딩.
    """
    n_ue = len(assign)

    # --- 용량 제약: capacity 초과 패턴에 위상 반전 ---
    # n_ue개 UE 중 capacity+1개 이상 선택된 모든 패턴을 마킹
    # 조합론적으로 (capacity+1)개 이상이 모두 1인 서브셋에 phase flip
    from itertools import combinations
    if n_ue > capacity:
        for combo in combinations(range(n_ue), capacity + 1):
            # combo에 포함되지 않는 큐비트에 X 적용 → MCX로 해당 패턴 감지
            others = [i for i in range(n_ue) if i not in combo]
            for i in others:
                qc.x(assign[i])
            qc.mcx(list(assign), flag[0])
            for i in others:
                qc.x(assign[i])

    # --- 목적함수: 채널 품질을 위상 회전으로 인코딩 ---
    max_q = float(np.max(qualities)) if np.max(qualities) > 0 else 1.0
    for i in range(n_ue):
        if qualities[i] > 0:
            theta = np.pi * (qualities[i] / max_q) * 0.5
            qc.rz(theta, assign[i])

    # --- 용량 제약 uncompute ---
    if n_ue > capacity:
        for combo in reversed(list(combinations(range(n_ue), capacity + 1))):
            others = [i for i in range(n_ue) if i not in combo]
            for i in others:
                qc.x(assign[i])
            qc.mcx(list(assign), flag[0])
            for i in others:
                qc.x(assign[i])


def solve_local(subproblem, shots=4096, iterations=1):
    """로컬 서브문제를 양자 회로로 풀고 결과 반환."""
    qc = build_local_grover(subproblem, iterations=iterations)

    backend = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # 최빈 결과를 할당으로 해석
    best_bitstring = max(counts, key=counts.get)
    # Qiskit 측정 비트스트링은 역순 (q[0]이 오른쪽)
    assignment = [int(b) for b in reversed(best_bitstring)]

    return {
        "subproblem": subproblem,
        "counts": counts,
        "best_bitstring": best_bitstring,
        "assignment": assignment,  # assignment[i] = 1이면 i번째 로컬 UE가 해당 AP에 할당
        "circuit": qc,
    }


# ============================================================
# Step 3: Classical Communication (고전 통신으로 경계 정보 교환)
# ============================================================

def exchange_boundary_info(local_results):
    """
    각 셀의 로컬 해를 수집하고 경계 UE에 대한 할당 정보를 교환.
    DQML CC scheme에서 영감: mid-circuit measurement 결과를 고전 비트로 전달.
    (이 예제에서는 post-measurement 기반으로 단순화)
    """
    # 글로벌 할당 테이블: UE → [(AP, 할당여부)]
    global_assignment = {}

    for res in local_results:
        sub = res["subproblem"]
        ap = sub["ap_index"]
        for local_idx, ue_global in enumerate(sub["ue_indices"]):
            assigned = res["assignment"][local_idx]
            if ue_global not in global_assignment:
                global_assignment[ue_global] = []
            global_assignment[ue_global].append({
                "ap": ap,
                "assigned": assigned,
                "quality": sub["qualities"][local_idx],
                "cell": sub["name"],
            })

    return global_assignment


# ============================================================
# Step 4: Reconciliation (경계 충돌 해소)
# ============================================================

def reconcile_conflicts(global_assignment):
    """
    경계 UE가 여러 셀에서 동시 할당된 경우 충돌 해소.
    전략: 채널 품질이 더 높은 AP에 할당 (greedy classical).
    """
    final_assignment = {}  # UE → AP (or None)
    conflicts = []
    ap_load = {0: 0, 1: 0, 2: 0}  # 각 AP의 현재 부하

    # 먼저 비경계 UE 처리 (충돌 없음)
    for ue, entries in sorted(global_assignment.items()):
        if ue not in BOUNDARY_UES:
            for e in entries:
                if e["assigned"]:
                    final_assignment[ue] = e["ap"]
                    ap_load[e["ap"]] += 1
                    break
            else:
                final_assignment[ue] = None

    # 경계 UE 충돌 해소
    for ue in sorted(BOUNDARY_UES):
        entries = global_assignment.get(ue, [])
        assigned_entries = [e for e in entries if e["assigned"]]

        if len(assigned_entries) == 0:
            final_assignment[ue] = None
            conflicts.append((ue, "unassigned", None))
        elif len(assigned_entries) == 1:
            chosen = assigned_entries[0]
            if ap_load[chosen["ap"]] < AP_CAPACITY:
                final_assignment[ue] = chosen["ap"]
                ap_load[chosen["ap"]] += 1
            else:
                final_assignment[ue] = None
                conflicts.append((ue, "capacity_exceeded", chosen["ap"]))
        else:
            # 충돌! 여러 셀에서 동시 할당 → 품질 기준 선택
            conflicts.append((ue, "multi_assign", [e["ap"] for e in assigned_entries]))
            # 용량 여유가 있고 품질 최고인 AP 선택
            best = None
            for e in sorted(assigned_entries, key=lambda x: -x["quality"]):
                if ap_load[e["ap"]] < AP_CAPACITY:
                    best = e
                    break
            if best:
                final_assignment[ue] = best["ap"]
                ap_load[best["ap"]] += 1
            else:
                final_assignment[ue] = None

    return final_assignment, conflicts, ap_load


# ============================================================
# Step 5: Benchmark — 중앙집중 vs 하이브리드 비교
# ============================================================

def centralized_optimal():
    """
    브루트포스로 글로벌 최적해 계산 (고전 기준선).
    각 UE를 연결 가능한 AP 중 하나에 할당 (또는 미할당).
    """
    best_score = -1
    best_assign = None

    # 각 UE의 가능한 AP 목록 (-1 = 미할당 포함)
    ue_options = []
    for ue in range(N_UE):
        opts = [-1]  # 미할당
        for ap in range(N_AP):
            if LINK_QUALITY[ue, ap] > 0:
                opts.append(ap)
        ue_options.append(opts)

    for combo in product(*ue_options):
        # 용량 체크
        ap_counts = [0] * N_AP
        valid = True
        score = 0.0
        for ue, ap in enumerate(combo):
            if ap >= 0:
                ap_counts[ap] += 1
                if ap_counts[ap] > AP_CAPACITY:
                    valid = False
                    break
                score += LINK_QUALITY[ue, ap]
        if valid and score > best_score:
            best_score = score
            best_assign = combo

    return best_assign, best_score


def compute_hybrid_score(final_assignment):
    """하이브리드 결과의 총 품질 점수 계산."""
    score = 0.0
    for ue, ap in final_assignment.items():
        if ap is not None:
            score += LINK_QUALITY[ue, ap]
    return score


# ============================================================
# Main: 전체 파이프라인 실행
# ============================================================

def main():
    print("=" * 65)
    print("  Hybrid Quantum-Classical Network Assignment")
    print("  6 UE × 3 AP, AP capacity = 2, 경계 UE: {2, 4}")
    print("=" * 65)

    # ----- Step 1: 네트워크 분할 -----
    print("\n[Step 1] Network Decomposition")
    subproblems = decompose_network()
    for sp in subproblems:
        print(f"  {sp['name']}: UE{sp['ue_indices']} → AP{sp['ap_index']}, "
              f"qualities={sp['qualities']}")

    # ----- Step 2: 로컬 양자 솔버 -----
    print("\n[Step 2] Local Quantum Solver (Grover)")
    local_results = []
    for sp in subproblems:
        res = solve_local(sp, shots=4096, iterations=2)
        local_results.append(res)
        ue_names = [f"UE{u}" for u in sp["ue_indices"]]
        assign_str = ", ".join(
            f"{name}={'O' if a else 'X'}"
            for name, a in zip(ue_names, res["assignment"])
        )
        print(f"  {sp['name']}: {assign_str}")
        # 상위 5개 측정 결과
        sorted_counts = sorted(res["counts"].items(), key=lambda x: -x[1])[:5]
        print(f"    Top counts: {dict(sorted_counts)}")

    # ----- Step 3: 고전 통신 -----
    print("\n[Step 3] Classical Communication (경계 정보 교환)")
    global_assignment = exchange_boundary_info(local_results)
    for ue in sorted(global_assignment.keys()):
        entries = global_assignment[ue]
        marker = " ← boundary" if ue in BOUNDARY_UES else ""
        info = ", ".join(
            f"AP{e['ap']}({'할당' if e['assigned'] else '미할당'}, q={e['quality']:.1f})"
            for e in entries
        )
        print(f"  UE{ue}: {info}{marker}")

    # ----- Step 4: 충돌 해소 -----
    print("\n[Step 4] Reconciliation (충돌 해소)")
    final, conflicts, ap_load = reconcile_conflicts(global_assignment)
    if conflicts:
        for ue, ctype, detail in conflicts:
            print(f"  충돌 감지: UE{ue} — {ctype} (detail={detail})")
    else:
        print("  충돌 없음")

    print("\n  최종 할당:")
    for ue in sorted(final.keys()):
        ap = final[ue]
        if ap is not None:
            q = LINK_QUALITY[ue, ap]
            print(f"    UE{ue} → AP{ap} (quality={q:.1f})")
        else:
            print(f"    UE{ue} → 미할당")
    print(f"  AP 부하: {dict(ap_load)}")

    # ----- Step 5: 벤치마크 -----
    print("\n[Step 5] Benchmark")
    hybrid_score = compute_hybrid_score(final)

    opt_assign, opt_score = centralized_optimal()
    print(f"  중앙집중 최적해: {opt_assign}, 총 품질 = {opt_score:.1f}")
    print(f"  하이브리드 결과:  총 품질 = {hybrid_score:.1f}")
    if opt_score > 0:
        ratio = hybrid_score / opt_score * 100
        print(f"  성능비: {ratio:.1f}%")

    # ----- 회로 정보 출력 -----
    print("\n[Circuit Info]")
    for res in local_results:
        qc = res["circuit"]
        print(f"  {res['subproblem']['name']}: "
              f"{qc.num_qubits} qubits, depth={qc.depth()}")

    print("\n" + "=" * 65)
    print("  완료: 분산 양자 → 고전 통신 → 충돌 해소 파이프라인 시연")
    print("=" * 65)


if __name__ == "__main__":
    main()
