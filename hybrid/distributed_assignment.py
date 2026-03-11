#!/usr/bin/env python3
"""
distributed_assignment.py
=========================
DQNA — Distributed Quantum Network Assignment

경계 UE를 여러 zone에 복제하여 각 QPU가 독립·병렬 실행,
측정 후 고전 조정(reconciliation)으로 글로벌 할당 확정.

Architecture
------------
1. Partition : 글로벌 네트워크 → zone 단위 서브문제 분할
2. Zone Solve: 각 QPU가 선택 문제(assign/not)를 Grover 회로로 해결
3. Collect   : 경계 UE의 다중 후보 수집
4. Resolve   : 품질·용량 기반 충돌 해소 → 최종 할당
5. Compare   : brute-force 최적해와 비교

Scenario
--------
  6 UE × 3 AP,  AP 용량 = 2
  Zone 0 (AP0): UE0, UE1, UE2*      (* = boundary)
  Zone 1 (AP1): UE2*, UE3, UE4*
  Zone 2 (AP2): UE4*, UE5

Author: Yong Hun Jang  |  Korea University
"""

import numpy as np
from dataclasses import dataclass
from collections import Counter
from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector


# ────────────────────────────────────────────────────────────
#  Data Model
# ────────────────────────────────────────────────────────────

@dataclass
class Network:
    """글로벌 무선 네트워크."""
    n_ue: int
    n_ap: int
    quality: np.ndarray       # (n_ue, n_ap),  -1 = 링크 없음
    capacity: list[int]       # AP별 최대 UE 수

    def score(self, assignment: dict) -> float:
        """총 링크 품질.  assignment = {ue: ap} (None = 미할당)."""
        return sum(
            self.quality[u, a]
            for u, a in assignment.items()
            if a is not None and self.quality[u, a] > 0
        )

    def feasible(self, assignment: dict) -> bool:
        """AP 용량 제약 검사."""
        counts = Counter(a for a in assignment.values() if a is not None)
        return all(counts.get(j, 0) <= self.capacity[j]
                   for j in range(self.n_ap))


@dataclass
class Zone:
    """한 zone의 로컬 서브문제."""
    zone_id: int
    ap: int                   # 이 zone이 담당하는 AP
    ues: list[int]            # 글로벌 UE 인덱스 (경계 UE 포함)
    qualities: np.ndarray     # 로컬 UE → 이 AP 링크 품질
    cap: int                  # AP 용량
    boundary: list[bool]      # boundary[i] = True → 경계 UE


# ────────────────────────────────────────────────────────────
#  1. Partition
# ────────────────────────────────────────────────────────────

def partition(net: Network,
              zone_defs: list[dict]) -> list[Zone]:
    """
    네트워크를 zone 단위로 분할.
    zone_defs: [{"ap": int, "ues": [int]}, ...]
    경계 UE = 2개 이상 zone에 등장하는 UE.
    """
    ue_freq = Counter()
    for zd in zone_defs:
        for u in zd["ues"]:
            ue_freq[u] += 1

    zones = []
    for i, zd in enumerate(zone_defs):
        ap = zd["ap"]
        ues = zd["ues"]
        quals = np.array([max(0.0, net.quality[u, ap]) for u in ues])
        zones.append(Zone(
            zone_id=i,
            ap=ap,
            ues=ues,
            qualities=quals,
            cap=net.capacity[ap],
            boundary=[ue_freq[u] > 1 for u in ues],
        ))
    return zones


# ────────────────────────────────────────────────────────────
#  2. Quantum Oracle
# ────────────────────────────────────────────────────────────

def _violation_compute(qc, assign, flag, cap, min_assign=1):
    """
    비실현 가능 패턴에 대해 flag 토글.
    비실현 = hamming weight > cap  OR  hamming weight < min_assign.
    각 비트 패턴마다 MCX 적용 — 상호 배타적이므로
    주어진 basis state에서 최대 1회만 발화.
    """
    n = len(assign)
    for pat in range(1 << n):
        hw = bin(pat).count('1')
        if hw > cap or hw < min_assign:
            bits = format(pat, f'0{n}b')
            flips = [i for i, b in enumerate(bits) if b == '0']
            for i in flips:
                qc.x(assign[i])
            qc.mcx(list(assign), flag[0])
            for i in reversed(flips):
                qc.x(assign[i])


def _quality_angles(quals, alpha=0.3):
    """
    품질 → Ry 초기 회전각 계산.
    θ = π/2 → 균일 (50/50),  θ > π/2 → |1⟩ 쪽 편향 (할당 선호).
    alpha: 편향 강도.  0 → 균일,  1 → 최대 편향.
    """
    qmax = float(np.max(quals)) if np.max(quals) > 0 else 1.0
    thetas = []
    for q in quals:
        if q > 0:
            thetas.append(np.pi / 2 + alpha * (np.pi / 2) * (q / qmax))
        else:
            thetas.append(np.pi / 2)
    return thetas


def _prepare_biased(qc, assign, thetas):
    """A|0⟩: 품질 기반 biased initial state."""
    for i, theta in enumerate(thetas):
        qc.ry(theta, assign[i])


def _unprepare_biased(qc, assign, thetas):
    """A†: biased state 역연산."""
    for i, theta in enumerate(thetas):
        qc.ry(-theta, assign[i])


def _biased_diffusion(qc, assign, thetas):
    """
    Biased Grover diffusion:  A (2|0⟩⟨0| - I) A†
    A = ⊗ Ry(θ_i),  품질 가중 초기 상태 기준 반사.
    """
    n = len(assign)
    _unprepare_biased(qc, assign, thetas)
    qc.x(assign)
    qc.h(assign[-1])
    if n > 1:
        qc.mcx(list(assign[:-1]), assign[-1])
    qc.h(assign[-1])
    qc.x(assign)
    _prepare_biased(qc, assign, thetas)


# ────────────────────────────────────────────────────────────
#  3. Zone Circuit
# ────────────────────────────────────────────────────────────

def build_zone_circuit(zone: Zone,
                       grover_iter: int = 2,
                       alpha: float = 0.3,
                       min_assign: int = 1) -> QuantumCircuit:
    """
    Zone 선택 회로.  qubit[i] = |1⟩ → UE i를 이 AP에 할당.

    설계:
      품질 인코딩: biased initial state (Ry per qubit, 고품질 → |1⟩ 편향)
      제약 oracle:  feasible 상태에 -1 위상 (phase kickback)
      Diffusion:    biased state 기준 반사 → 고품질 feasible 상태 증폭

    분리 원칙: 품질은 초기상태, 제약은 oracle, 간섭 없음.
    """
    n = len(zone.ues)

    q   = QuantumRegister(n, "q")
    vf  = QuantumRegister(1, "vf")    # violation flag
    sf  = QuantumRegister(1, "sf")    # superflag (|−⟩)
    cr  = ClassicalRegister(n, "c")
    qc  = QuantumCircuit(q, vf, sf, cr)

    # ── 초기화: biased state + superflag ──
    thetas = _quality_angles(zone.qualities, alpha)
    _prepare_biased(qc, q, thetas)
    qc.x(sf)
    qc.h(sf)          # |−⟩ for phase kickback
    qc.barrier()

    for _ in range(grover_iter):
        # ── Oracle: feasible → phase -1 ──
        _violation_compute(qc, q, vf, zone.cap, min_assign)
        qc.x(vf)                                     # vf=|1⟩ if feasible
        qc.cx(vf, sf)                                # feasible → phase -1
        qc.x(vf)                                     # restore
        _violation_compute(qc, q, vf, zone.cap, min_assign)
        qc.barrier()

        # ── Biased Diffusion ──
        _biased_diffusion(qc, q, thetas)
        qc.barrier()

    qc.measure(q, cr)
    return qc


# ────────────────────────────────────────────────────────────
#  4. Execution
# ────────────────────────────────────────────────────────────

def run_zone(zone: Zone, shots: int = 4096,
             grover_iter: int = 2, alpha: float = 0.3,
             min_assign: int = 1) -> dict:
    """한 zone 회로 실행 → 측정 결과 반환."""
    qc = build_zone_circuit(zone, grover_iter, alpha, min_assign)
    backend = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc, backend)
    counts = backend.run(t_qc, shots=shots).result().get_counts()

    best = max(counts, key=counts.get)
    selection = [int(b) for b in reversed(best)]   # q[0]부터

    return {
        "zone": zone,
        "counts": counts,
        "best": best,
        "selection": selection,
        "circuit": qc,
    }


def run_all_zones(zones: list[Zone], shots: int = 4096,
                  grover_iter: int = 2, alpha: float = 0.3,
                  min_assign: int = 1) -> list[dict]:
    """모든 zone 독립 실행 (실제로는 병렬 QPU)."""
    return [run_zone(z, shots, grover_iter, alpha, min_assign) for z in zones]


# ────────────────────────────────────────────────────────────
#  5. Reconciliation
# ────────────────────────────────────────────────────────────

def collect_candidates(zone_results: list[dict]) -> dict:
    """
    Zone별 측정 결과 → UE별 후보 수집.
    Returns {ue: [{"ap", "quality", "zone", "selected"}, ...]}
    """
    cands = {}
    for res in zone_results:
        z = res["zone"]
        for i, ue in enumerate(z.ues):
            entry = {
                "ap": z.ap,
                "quality": float(z.qualities[i]),
                "zone": z.zone_id,
                "selected": bool(res["selection"][i]),
            }
            cands.setdefault(ue, []).append(entry)
    return cands


def resolve_boundaries(cands: dict, net: Network) -> dict:
    """
    경계 UE 충돌 해소 → 최종 할당 {ue: ap or None}.
    전략: selected 후보 중 품질 최고 + 용량 여유 있는 AP 선택.
    """
    final = {}
    ap_load = Counter()

    # 1차: 비경계 UE (후보 1개)
    for ue in sorted(cands):
        entries = cands[ue]
        if len(entries) == 1:
            e = entries[0]
            if e["selected"] and ap_load[e["ap"]] < net.capacity[e["ap"]]:
                final[ue] = e["ap"]
                ap_load[e["ap"]] += 1
            else:
                final[ue] = None

    # 2차: 경계 UE (후보 2개+)
    for ue in sorted(cands):
        entries = cands[ue]
        if len(entries) <= 1:
            continue
        selected = [e for e in entries if e["selected"]]
        if not selected:
            final[ue] = None
            continue
        # 품질 내림차순 → 용량 여유 확인
        for e in sorted(selected, key=lambda x: -x["quality"]):
            if ap_load[e["ap"]] < net.capacity[e["ap"]]:
                final[ue] = e["ap"]
                ap_load[e["ap"]] += 1
                break
        else:
            final[ue] = None

    return final


def run_pipeline(net: Network, zones: list[Zone],
                 shots: int = 4096, grover_iter: int = 2,
                 alpha: float = 0.3, min_assign: int = 1,
                 rounds: int = 10, top_k: int = 3,
                 verbose: bool = False) -> dict:
    """
    다중 라운드 파이프라인.

    각 round에서:
      1) 모든 zone 독립 실행
      2) zone별 top-K 측정 결과의 모든 조합을 평가
      3) feasible + 최고 점수 조합을 best-so-far로 갱신

    Returns dict: assignment, score, feasible, rounds_run, history
    """
    best_assign = None
    best_score = -1.0
    history = []

    for r in range(rounds):
        results = run_all_zones(zones, shots, grover_iter, alpha, min_assign)

        # zone별 top-K 선택지 추출
        zone_options = []
        for res in results:
            z = res["zone"]
            n_local = len(z.ues)
            sorted_counts = sorted(res["counts"].items(),
                                   key=lambda x: -x[1])[:top_k]
            options = []
            for bs, _ in sorted_counts:
                sel = [int(b) for b in reversed(bs)]
                options.append(sel)
            zone_options.append((z, options))

        # 모든 조합 평가
        from itertools import product as iprod
        option_lists = [opts for _, opts in zone_options]
        for combo in iprod(*option_lists):
            # combo[zone_idx] = selection list for that zone
            cands = {}
            for zone_idx, sel in enumerate(combo):
                z = zone_options[zone_idx][0]
                for i, ue in enumerate(z.ues):
                    entry = {
                        "ap": z.ap,
                        "quality": float(z.qualities[i]),
                        "zone": z.zone_id,
                        "selected": bool(sel[i]),
                    }
                    cands.setdefault(ue, []).append(entry)

            final = resolve_boundaries(cands, net)
            if not net.feasible(final):
                continue
            score = net.score(final)
            if score > best_score:
                best_score = score
                best_assign = dict(final)

        history.append({"round": r + 1, "best_score": best_score})
        if verbose:
            print(f"  Round {r+1}/{rounds}: best_score={best_score:.1f}")

    return {
        "assignment": best_assign,
        "score": best_score,
        "feasible": net.feasible(best_assign) if best_assign else False,
        "rounds_run": rounds,
        "history": history,
    }


# ────────────────────────────────────────────────────────────
#  6. Classical Baseline
# ────────────────────────────────────────────────────────────

def brute_force(net: Network) -> tuple:
    """Exhaustive search.  Returns ({ue: ap}, score)."""
    options = []
    for u in range(net.n_ue):
        opts = [None]
        for a in range(net.n_ap):
            if net.quality[u, a] > 0:
                opts.append(a)
        options.append(opts)

    best_assign, best_score = None, -1.0
    for combo in product(*options):
        assign = dict(enumerate(combo))
        if net.feasible(assign):
            s = net.score(assign)
            if s > best_score:
                best_score = s
                best_assign = assign
    return best_assign, best_score


# ────────────────────────────────────────────────────────────
#  7. Statevector Debugging
# ────────────────────────────────────────────────────────────

def analyze_statevector(zone: Zone,
                        grover_iter: int = 2,
                        alpha: float = 0.3,
                        min_assign: int = 1):
    """측정 전 상태벡터 확률 분석 (oracle 동작 검증용)."""
    qc = build_zone_circuit(zone, grover_iter, alpha, min_assign)
    qc_nm = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_nm)
    probs = sv.probabilities_dict()

    n = len(zone.ues)
    assign_probs = {}
    for bs, p in probs.items():
        key = bs[-n:]
        assign_probs[key] = assign_probs.get(key, 0) + p

    print(f"\n  Zone {zone.zone_id} (AP{zone.ap}, UE{zone.ues}):")
    print(f"    qualities = {zone.qualities}")
    for bs, p in sorted(assign_probs.items(), key=lambda x: -x[1])[:8]:
        sel = [int(b) for b in reversed(bs)]
        hw = sum(sel)
        feas = "V" if (1 <= hw <= zone.cap) else "X"
        assigned = [f"UE{zone.ues[i]}" for i, s in enumerate(sel) if s]
        print(f"    |{bs}> P={p:.4f} hw={hw} {feas} {assigned}")


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────

def main():
    # ── 네트워크 정의 ──
    quality = np.array([
        [ 9.0, -1.0, -1.0],   # UE0: Zone 0 전용
        [ 7.5, -1.0, -1.0],   # UE1: Zone 0 전용
        [ 4.0,  6.0, -1.0],   # UE2: 경계 (Zone 0/1)
        [-1.0,  8.0, -1.0],   # UE3: Zone 1 전용
        [-1.0,  5.0,  7.0],   # UE4: 경계 (Zone 1/2)
        [-1.0, -1.0,  8.5],   # UE5: Zone 2 전용
    ])

    net = Network(
        n_ue=6, n_ap=3,
        quality=quality,
        capacity=[2, 2, 2],
    )

    zone_defs = [
        {"ap": 0, "ues": [0, 1, 2]},
        {"ap": 1, "ues": [2, 3, 4]},
        {"ap": 2, "ues": [4, 5]},
    ]

    print("=" * 60)
    print("  DQNA - Distributed Quantum Network Assignment")
    print("  6 UE x 3 AP, capacity=2, boundary={UE2, UE4}")
    print("=" * 60)

    # ── 1. Partition ──
    zones = partition(net, zone_defs)
    print("\n[1] Partition")
    for z in zones:
        bnd = [z.ues[i] for i, b in enumerate(z.boundary) if b]
        print(f"  Zone {z.zone_id} (AP{z.ap}): UE{z.ues}, "
              f"q={z.qualities}, boundary={bnd}")

    # ── 2. Statevector ──
    print("\n[2] Statevector Analysis (oracle verification)")
    for z in zones:
        analyze_statevector(z, grover_iter=1, alpha=0.3, min_assign=1)

    # ── 3. Single-round execution ──
    print("\n[3] Single-Round Execution (independent QPUs)")
    results = run_all_zones(zones, shots=4096, grover_iter=1,
                            alpha=0.3, min_assign=1)
    for res in results:
        z = res["zone"]
        names = [f"UE{u}" for u in z.ues]
        sel_str = ", ".join(
            f"{nm}={'O' if s else 'X'}"
            for nm, s in zip(names, res["selection"])
        )
        top = sorted(res["counts"].items(), key=lambda x: -x[1])[:5]
        print(f"  Zone {z.zone_id}: {sel_str}")
        print(f"    Top: {dict(top)}")

    # ── 4. Single-round reconciliation ──
    print("\n[4] Single-Round Reconciliation")
    cands = collect_candidates(results)
    for ue in sorted(cands):
        entries = cands[ue]
        bnd = " <- boundary" if len(entries) > 1 else ""
        info = ", ".join(
            f"AP{e['ap']}({'O' if e['selected'] else 'X'} q={e['quality']:.1f})"
            for e in entries
        )
        print(f"  UE{ue}: {info}{bnd}")

    final_single = resolve_boundaries(cands, net)
    print("\n  Final (single round):")
    for ue in sorted(final_single):
        ap = final_single[ue]
        if ap is not None:
            print(f"    UE{ue} -> AP{ap} (q={net.quality[ue, ap]:.1f})")
        else:
            print(f"    UE{ue} -> unassigned")
    single_score = net.score(final_single)

    # ── 5. Multi-round pipeline ──
    print("\n[5] Multi-Round Pipeline (10 rounds, top-3 combinations)")
    pipe = run_pipeline(net, zones, shots=4096, grover_iter=1,
                        alpha=0.3, min_assign=1,
                        rounds=10, top_k=3, verbose=True)

    print(f"\n  Best assignment:")
    if pipe["assignment"]:
        for ue in sorted(pipe["assignment"]):
            ap = pipe["assignment"][ue]
            if ap is not None:
                print(f"    UE{ue} -> AP{ap} (q={net.quality[ue, ap]:.1f})")
            else:
                print(f"    UE{ue} -> unassigned")

    # ── 6. Benchmark ──
    print("\n[6] Benchmark")
    opt_assign, opt_score = brute_force(net)
    print(f"  {'Method':<25} {'Score':>8} {'Ratio':>8} {'Feasible':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'Classical optimal':<25} {opt_score:8.1f} {'--':>8} {'True':>10}")
    print(f"  {'DQNA (single round)':<25} {single_score:8.1f} "
          f"{single_score/opt_score*100:7.1f}% "
          f"{str(net.feasible(final_single)):>10}")
    print(f"  {'DQNA (multi-round)':<25} {pipe['score']:8.1f} "
          f"{pipe['score']/opt_score*100:7.1f}% "
          f"{str(pipe['feasible']):>10}")

    # ── Circuit info ──
    print("\n[Circuit Info]")
    for res in results:
        z = res["zone"]
        qc = res["circuit"]
        print(f"  Zone {z.zone_id}: {qc.num_qubits} qubits, depth={qc.depth()}")

    print("\n" + "=" * 60)
    print("  Done: parallel QPUs -> boundary resolve -> global assignment")
    print("=" * 60)


if __name__ == "__main__":
    main()
