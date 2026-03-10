#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# 4 UE x (aBS, bBS, cBS, idle)
# Encoding (per UE, 2 qubits):
#   00 -> aBS, 01 -> bBS, 10 -> cBS, 11 -> idle
#
# Weight maximization (soft):
#   cost qubits start at |0>
#   theta = alpha*pi*w_norm  -> P(cost=1)=sin^2(theta/2)
#   so max weight -> theta ~ pi -> cost=1 prob ~ 1
#
# Hardcoded feasibility (36 states):
#   - no idle
#   - all 3 BS used at least once
#   - each BS <= 2 users
#   => exactly (2,1,1) distribution -> 36 assignments with UE IDs
# ============================================================

import numpy as np
from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram


# In[2]:


# -----------------------------
# Problem size / encoding
# -----------------------------
N_UE = 4
BITS_PER_UE = 2
N_STATE = N_UE * BITS_PER_UE      # 8 assignment qubits
N_COST  = N_UE                    # 4 cost qubits

CODES = ["00", "01", "10", "11"]  # a, b, c, idle
CODE2NAME = {"00": "aBS", "01": "bBS", "10": "cBS", "11": "idle"}

BS_CODES_ONLY = ["00", "01", "10"]  # feasible set uses only a,b,c (idle excluded)


# In[3]:


# -----------------------------
# Feasible patterns: exactly 36
# (2,1,1) distribution over a,b,c with UE IDs
# -----------------------------
def generate_feasible_patterns_36():
    patterns = []
    for choice in product(BS_CODES_ONLY, repeat=N_UE):  # tuple length 4
        # all BS used at least once, and each used <=2
        if all(choice.count(c) >= 1 for c in BS_CODES_ONLY) and all(choice.count(c) <= 2 for c in BS_CODES_ONLY):
            patterns.append(list(choice))
    assert len(patterns) == 36, f"Expected 36 patterns, got {len(patterns)}"
    return patterns

FEASIBLE_36 = generate_feasible_patterns_36()


# In[4]:


# -----------------------------
# State preparation
# -----------------------------
def prepare_state_register(qc, assign, cost, anc_cost, superflag):
    """
    assign: uniform superposition
    cost: |0> for all UE  (maximization mapping)
    anc_cost: |0>
    superflag: |-> for phase kickback
    """
    qc.reset(assign)
    qc.h(assign)

    qc.reset(cost)         # keep at |0>
    qc.reset(anc_cost)

    qc.reset(superflag)
    qc.x(superflag)
    qc.h(superflag)

    qc.barrier()  # INIT


# In[5]:


# -----------------------------
# Per-UE match (used by weight oracle)
# -----------------------------
def compute_match_code(qc, assign, ue_idx, code_str, anc_cost):
    q0 = assign[2 * ue_idx + 0]
    q1 = assign[2 * ue_idx + 1]

    if code_str[0] == "0":
        qc.x(q0)
    if code_str[1] == "0":
        qc.x(q1)

    qc.mcx([q0, q1], anc_cost[0])  # toggle


def uncompute_match_code(qc, assign, ue_idx, code_str, anc_cost):
    q0 = assign[2 * ue_idx + 0]
    q1 = assign[2 * ue_idx + 1]

    qc.mcx([q0, q1], anc_cost[0])

    if code_str[1] == "0":
        qc.x(q1)
    if code_str[0] == "0":
        qc.x(q0)


# In[6]:


# -----------------------------
# Full-pattern match on assign (8 qubits) using anc_cost (no extra qubits)
# anc_cost toggles iff assign matches the 4 code strings
# -----------------------------
def toggle_if_assign_matches_pattern(qc, assign, pattern_codes, anc_cost):
    """
    pattern_codes: list length 4, each in {"00","01","10"} (idle not used here)
    anc_cost[0] toggles iff all 8 assign bits match.
    """
    masked = []
    for ue_idx, code in enumerate(pattern_codes):
        for b in range(2):
            q = assign[2 * ue_idx + b]
            if code[b] == "0":
                qc.x(q)
                masked.append(q)

    # all 8 assignment qubits as controls
    qc.mcx(list(assign), anc_cost[0], mode="noancilla")

    for q in reversed(masked):
        qc.x(q)


# In[7]:


# -----------------------------
# Objective oracle (weight only)
# maximization mapping: theta in [0, pi]
# -----------------------------
def oracle_objective_phase(qc, assign, cost, anc_cost, W, alpha=1.0):
    W = np.asarray(W, dtype=float)
    w_min, w_max = float(np.min(W)), float(np.max(W))
    if w_max - w_min <= 1e-12:
        qc.barrier()
        return

    for i in range(N_UE):
        for j, code in enumerate(CODES):
            w_norm = (W[i, j] - w_min) / (w_max - w_min)   # 0..1
            theta  = alpha * np.pi * w_norm                # 0..alpha*pi

            compute_match_code(qc, assign, i, code, anc_cost)
            qc.cry(theta, anc_cost[0], cost[i])
            uncompute_match_code(qc, assign, i, code, anc_cost)

    qc.barrier()  # WEIGHT_ORACLE


def inverse_oracle_objective_phase(qc, assign, cost, anc_cost, W, alpha=1.0):
    W = np.asarray(W, dtype=float)
    w_min, w_max = float(np.min(W)), float(np.max(W))
    if w_max - w_min <= 1e-12:
        qc.barrier()
        return

    # inverse: same terms with -theta (order not critical here)
    for i in range(N_UE):
        for j, code in enumerate(CODES):
            w_norm = (W[i, j] - w_min) / (w_max - w_min)   # 0..1
            theta  = alpha * np.pi * w_norm

            compute_match_code(qc, assign, i, code, anc_cost)
            qc.cry(-theta, anc_cost[0], cost[i])
            uncompute_match_code(qc, assign, i, code, anc_cost)

    qc.barrier()  # UNCOMPUTE_WEIGHT


# In[8]:


# -----------------------------
# Hardcoded feasibility marking: only 36 assignments are allowed
# Mark condition = (assign in FEASIBLE_36) AND (all cost == 1)
#
# We do NOT add any new qubits:
#   - reuse anc_cost as a temporary "pattern matched" flag
# -----------------------------
def apply_superflag_mark_36(qc, assign, cost, anc_cost, superflag, patterns_36):
    qc.barrier()  # FEAS_36_START

    for pattern in patterns_36:
        # anc_cost ^= 1 iff assign matches this pattern
        toggle_if_assign_matches_pattern(qc, assign, pattern, anc_cost)

        # If (all cost==1) AND (anc_cost==1) -> flip superflag (|-> gives phase kickback)
        qc.mcx(list(cost) + [anc_cost[0]], superflag[0], mode="noancilla")

        # uncompute anc_cost (toggle back)
        toggle_if_assign_matches_pattern(qc, assign, pattern, anc_cost)

    qc.barrier()  # FEAS_36_END


# In[9]:


# -----------------------------
# Full oracle: (encode weight) -> (mark feasible & good) -> (uncompute)
# -----------------------------
def apply_oracle_weight_and_feasible36(qc, assign, cost, anc_cost, superflag, W, alpha=1.0):
    oracle_objective_phase(qc, assign, cost, anc_cost, W, alpha=alpha)
    apply_superflag_mark_36(qc, assign, cost, anc_cost, superflag, FEASIBLE_36)
    inverse_oracle_objective_phase(qc, assign, cost, anc_cost, W, alpha=alpha)


# In[10]:


# -----------------------------
# Diffusion on assign register
# -----------------------------
def apply_diffusion(qc, qubits):
    qc.barrier()  # DIFF_START

    qc.h(qubits)
    qc.x(qubits)

    qc.h(qubits[-1])
    qc.mcx(list(qubits[:-1]), qubits[-1], mode="noancilla")
    qc.h(qubits[-1])

    qc.x(qubits)
    qc.h(qubits)

    qc.barrier()  # DIFF_END


# In[11]:


# -----------------------------
# Decoding / scoring
# -----------------------------
def decode_assignment(bitstring, n_ue=4):
    bits = bitstring[::-1]
    out = []
    for i in range(n_ue):
        code = bits[2 * i] + bits[2 * i + 1]
        out.append(CODE2NAME.get(code, f"UNK({code})"))
    return out

def score_assignment(choices, W):
    name2col = {"aBS": 0, "bBS": 1, "cBS": 2, "idle": 3}
    return float(sum(W[i, name2col[ch]] for i, ch in enumerate(choices)))


# In[12]:


# ============================================================
# Example weights
# W[i, :] = [aBS, bBS, cBS, idle]
# idle은 feasibility에서 어차피 제외되지만, weight에도 낮게 두는 편이 안전
# ============================================================
W = np.array([
    [0.20, 0.60, 0.40, 0.01],  # UE0
    [0.70, 0.30, 0.50, 0.01],  # UE1
    [0.40, 0.80, 0.20, 0.01],  # UE2
    [0.50, 0.40, 0.60, 0.01],  # UE3
], dtype=float)

alpha = 1.0
iterations = 1
shots = 1000


# In[13]:


# -----------------------------
# Build circuit (wire order: anc_cost above cost)
# -----------------------------
assign    = QuantumRegister(N_STATE, "assign")
anc_cost  = QuantumRegister(1,       "anc_cost")
cost      = QuantumRegister(N_COST,  "cost")
superflag = QuantumRegister(1,       "superflag")
c_assign  = ClassicalRegister(N_STATE, "c_assign")

qc = QuantumCircuit(assign, anc_cost, cost, superflag, c_assign)

prepare_state_register(qc, assign, cost, anc_cost, superflag)

for it in range(iterations):
    qc.barrier()  # ITER_START
    apply_oracle_weight_and_feasible36(qc, assign, cost, anc_cost, superflag, W, alpha=alpha)
    apply_diffusion(qc, assign)
    qc.barrier()  # ITER_END

qc.barrier()  # MEASURE
qc.measure(assign, c_assign)


# In[14]:


# # Circuit draw
# qc.draw(
#     output="mpl",
#     scale=1.0,
#     fold=48,
#     style={"backgroundcolor": "#EEEEEE"},
#     plot_barriers=True
# )


# In[15]:


# -----------------------------
# Run
# -----------------------------
backend = Aer.get_backend("aer_simulator")
result = backend.run(qc, shots=shots).result()
counts = result.get_counts()

# plot_histogram(counts)


# In[16]:


# -----------------------------
# Inspect top results
# -----------------------------
#top_k = 10
#print(f"\nTop {top_k} measured assignments (decoded):")
#for bitstr, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
#    choices = decode_assignment(bitstr, n_ue=N_UE)
#    sc = score_assignment(choices, W)
#    print(f"count={cnt:5d} | bits={bitstr} | choices={choices} | score(sumW)={sc:.4f}")
    
import json

top_k = 10

items = []
for bitstr, cnt in counts.items():
    choices = decode_assignment(bitstr, n_ue=N_UE)
    sc = float(score_assignment(choices, W))
    items.append((bitstr, int(cnt), choices, sc))

# count ↓, score ↓, bits ↑(tie-break)
items.sort(key=lambda x: (-x[1], -x[3], x[0]))
top_items = items[:top_k]

top = [
    {"bits": b, "count": c, "choices": ch, "score": sc}
    for (b, c, ch, sc) in top_items
]

print(json.dumps({"top_k": top_k, "top": top}, ensure_ascii=False))



# In[ ]:




