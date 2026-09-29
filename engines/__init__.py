"""
EVolve Optimization Engines: Classical DP and Quantum QUBO Solvers
"""
from engines.classical_dp import solve as solve_classical_dp

try:
	from engines.quantum_qubo import solve as solve_quantum_qubo, build_qubo
except ModuleNotFoundError as error:
	if error.name not in {"neal", "dimod"}:
		raise

	def solve_quantum_qubo(*args, **kwargs):
		raise ModuleNotFoundError(
			"The QUBO solver requires the optional 'dwave-neal' and 'dimod' packages."
		) from None

	def build_qubo(*args, **kwargs):
		raise ModuleNotFoundError(
			"The QUBO solver requires the optional 'dwave-neal' and 'dimod' packages."
		) from None


__all__ = ["solve_classical_dp", "solve_quantum_qubo", "build_qubo"]
