Create a new module for the hybrid quantum-classical framework.
Module name: $ARGUMENTS

Requirements:
- Use Qiskit 1.1 API (QuantumCircuit, QuantumRegister, etc.)
- Follow existing conventions in CQF/ and QTG/ directories
- Include docstrings in English (Korean comments are OK for implementation details)
- Use QFT-based arithmetic (Draper adder) for any integer operations
- Enforce hard constraints via oracle design, NOT penalty/QUBO
- Include a test section at the bottom with a small example (e.g., 4UE/2AP)
- Print statevector results for verification
