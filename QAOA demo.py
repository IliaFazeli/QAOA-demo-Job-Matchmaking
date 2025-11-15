# QAOA demo: Job Matching (select K workers for a shift)

import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Qiskit imports (modern)
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Reproducibility
seed = 10
np.random.seed(seed)

# -------------------------
# STEP 1: Random dataset generator
# -------------------------
def generate_candidates(N):
    """
    Returns a list of dicts with fields: distance (0..1), skill (0..1), rating (0..1)
    """
    workers = []
    for i in range(N):
        d = np.clip(np.random.normal(0.3, 0.15), 0, 1)     # distance from job location
        s = np.clip(np.random.normal(0.5, 0.3), 0, 1)      # skill level matched to job    
        r = np.clip(np.random.normal(4.2, 0.6), 1, 5)      # rating out of 5 
        r = (r - 1) / 4.0   # normalize to 0..1
        workers.append({'id': i, 'distance': d, 'skill': s, 'rating': r})
    return workers

# -------------------------
# STEP 2: Score function & normalization
# -------------------------
def compute_scores(workers, alpha=1.0, beta=1.0, gamma=0.5): # Customizable context specific weights  
    """
    Compute per-worker score = alpha*skill + beta*rating - gamma*distance
    """
    raw = []
    for w in workers:
        score = alpha * w['skill'] + beta * w['rating'] - gamma * w['distance']
        raw.append(score)
    raw = np.array(raw)
    return raw

# -------------------------
# STEP 3: Build QUBO as SparsePauliOp (Modern Qiskit)
# -------------------------
def build_qubo(scores, K=2, penalty_lambda=None):
    """
    Build SparsePauliOp Hamiltonian implementing the QUBO:
      minimize -sum_i scores[i] * x_i  + lambda*(sum_i x_i - K)^2
    Returns SparsePauliOp, lambda used, and constant offset.
    """
    N = len(scores)
    
    # Calculate penalty lambda if not provided
    if penalty_lambda is None:
        max_score = max(abs(scores)) if len(scores) > 0 else 1.0
        penalty_lambda = max_score * 2.0 * K
    
    # Initialize Pauli list
    pauli_list = []
    
    # Linear coefficients h_i
    h = np.zeros(N)
    for i in range(N):
        h[i] = (-scores[i] / 2.0 + penalty_lambda * (1 - 2*K) / 2.0)
    
    # Quadratic coefficient J_ij (same for all pairs)
    J_ij = penalty_lambda / 2.0
    
    # Add single-qubit Z terms
    for i in range(N):
        if abs(h[i]) > 1e-10:
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_list.append((''.join(pauli_str), h[i]))
    
    # Add two-qubit ZZ terms
    for i, j in itertools.combinations(range(N), 2):
        if abs(J_ij) > 1e-10:
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_list.append((''.join(pauli_str), J_ij))
    
    # Calculate constant offset
    constant_offset = (
        -sum(scores) / 2.0 +
        penalty_lambda * (N + K**2) / 4.0
    )
    
    # Create SparsePauliOp
    if pauli_list:
        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
    else:
        cost_hamiltonian = SparsePauliOp(['I' * N], [0.0])
    
    return cost_hamiltonian, penalty_lambda, constant_offset


def bitstring_to_selection(bitstring):
    """Convert bitstring to list of selected indices"""
    return [i for i, bit in enumerate(bitstring) if bit == 1]


def evaluate_solution(bitstring, scores, K, penalty_lambda):
    """Evaluate QUBO objective for a bitstring"""
    x = np.array([int(b) for b in bitstring])
    score_term = -np.dot(scores, x)
    constraint_violation = np.sum(x) - K
    penalty_term = penalty_lambda * constraint_violation**2
    return score_term + penalty_term


# -------------------------
# STEP 4: Classical baselines
# -------------------------
def greedy_top_k(scores, K):
    """Pick K indices with highest scores"""
    inds = np.argsort(-scores)
    chosen = inds[:K]
    x = np.zeros_like(scores, dtype=int)
    x[chosen] = 1
    return x


def brute_force_best(scores, K):
    """Brute force best selection for small N"""
    N = len(scores)
    best_val = -1e9
    best_x = None
    for comb in itertools.combinations(range(N), K):
        x = np.zeros(N, dtype=int)
        x[list(comb)] = 1
        val = (scores * x).sum()
        if val > best_val:
            best_val = val
            best_x = x.copy()
    return best_x, best_val


# -------------------------
# STEP 5: QAOA solver (Modern Qiskit)
# -------------------------
def solve_qaoa(cost_hamiltonian, scores, K, penalty_lambda, backend=None, p=2, max_iter=100, shots=1024):
    """
    
    Args:
        cost_hamiltonian: SparsePauliOp representing the cost 
        scores: Original scores array
        K: Number of items (people) to select
        penalty_lambda: Penalty weight (calcuated during QUBO build)
        backend: Qiskit backend (uses AerSimulator if None)
        p: Number of QAOA layers (reps)
        max_iter: Maximum optimizer iterations
        shots: Number of shots per evaluation
    
    Returns:
        Selected binary vector x and full results dict
    """

    N = len(scores)
    
    # Use AerSimulator if no backend provided
    if backend is None:
        backend = AerSimulator()
    
    # Create QAOA ansatz
    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
    
    # Transpile for backend
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled_ansatz = pm.run(ansatz)
    
    # Initial parameters
    initial_params = np.random.uniform(0, 2*np.pi, ansatz.num_parameters)
    
    # Track optimization history
    objective_func_vals = []
    
    # Create Estimator for optimization
    estimator = Estimator()
    estimator.options.default_shots = shots
    
    # Cost function
    def cost_func_estimator(params):
        # Apply layout to Hamiltonian
        isa_hamiltonian = cost_hamiltonian.apply_layout(transpiled_ansatz.layout)
        
        # Run estimator
        pub = (transpiled_ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])
        
        results = job.result()[0]
        cost = results.data.evs
        
        objective_func_vals.append(cost)
        return cost
    
    # Optimize
    print(f"Starting QAOA optimization with p={p} layers...")
    result = minimize(
        cost_func_estimator,
        initial_params,
        method="COBYLA",
        options={'maxiter': max_iter}
    )
    
    print(f"Optimization complete: converged={result.success}, final_cost={result.fun:.4f}")
    
    # Sample optimized circuit
    optimized_ansatz = transpiled_ansatz.assign_parameters(result.x)
    optimized_ansatz.measure_all()
    
    sampler = Sampler()
    sampler.options.default_shots = shots * 10  # More shots for final sampling
    
    job = sampler.run([optimized_ansatz])
    counts = job.result()[0].data.meas.get_counts()
    
    # Find best solution
    best_cost = float('inf')
    best_bitstring = None
    
    for bitstring, count in counts.items():
        cost = evaluate_solution(bitstring, scores, K, penalty_lambda)
        if cost < best_cost:
            best_cost = cost
            best_bitstring = bitstring
    
    # Convert to binary vector
    x = np.array([int(b) for b in best_bitstring])
    
    return x, {
        'optimal_params': result.x,
        'optimal_cost': result.fun,
        'optimization_history': objective_func_vals,
        'counts': counts,
        'best_bitstring': best_bitstring,
        'converged': result.success
    }


# -------------------------
# STEP 6: Demonstration run function
# -------------------------
def demo(N, K):
    """Run complete demonstration"""
    print("=" * 70)
    print(f"QAOA Job Matching Demo: Select {K} workers from {N} candidates")
    print("=" * 70)
    
    # Generate workers
    workers = generate_candidates(N)
    scores = compute_scores(workers, alpha=1.0, beta=1.0, gamma=1.0)
    
    print("\nWorkers (id, distance, skill, rating, score):")
    print(f"{'ID':<4} {'Dist':<6} {'Skill':<6} {'Rating':<7} {'Score':<7}")
    print("-" * 40)
    for i, w in enumerate(workers):
        print(f"{i:<4} {w['distance']:<6.2f} {w['skill']:<6.2f} {w['rating']:<7.2f} {scores[i]:<7.3f}")
    
    # Build QUBO
    cost_hamiltonian, lam, offset = build_qubo(scores, K)
    print(f"\nPenalty lambda chosen: {lam:.2f}")
    print(f"Hamiltonian has {len(cost_hamiltonian)} Pauli terms")
    print(f"Constant offset: {offset:.4f}")
    
    # Classical greedy baseline
    print("\n" + "=" * 70)
    print("CLASSICAL BASELINES")
    print("=" * 70)
    
    greedy_x = greedy_top_k(scores, K)
    greedy_val = (scores * greedy_x).sum()
    print(f"\nGreedy selection: {np.where(greedy_x)[0].tolist()}")
    print(f"Greedy value: {greedy_val:.3f}")
    
    # Brute force (only for small N to keep reasonable runtime)
    if N <= 12:
        brute_x, brute_val = brute_force_best(scores, K)
        print(f"\nBrute-force best: {np.where(brute_x)[0].tolist()}")
        print(f"Brute-force value: {brute_val:.3f}")
    else:
        brute_x, brute_val = None, None
        print("\n(Skipping brute-force for N > 12)")
    
    # QAOA
    print("\n" + "=" * 70)
    print("QAOA SOLUTION")
    print("=" * 70)
    
    x_qaoa, qaoa_result = solve_qaoa(
        cost_hamiltonian, 
        scores, 
        K, 
        lam, 
        backend=None,  # Uses AerSimulator
        p=2,           # 2 QAOA layers
        max_iter=50,
        shots=1024
    )
    
    qaoa_val = (scores * x_qaoa).sum()
    selected_workers = np.where(x_qaoa)[0]
    num_selected = x_qaoa.sum()
    
    print(f"\nQAOA selection: {selected_workers.tolist()}")
    print(f"Number selected: {num_selected} (target: {K})")
    print(f"QAOA value: {qaoa_val:.3f}")
    print(f"Constraint satisfied: {num_selected == K}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Method':<15} {'Selection':<20} {'Value':<10} {'Gap from Best'}")
    print("-" * 70)
    
    best_val = brute_val if brute_val is not None else greedy_val
    
    print(f"{'Greedy':<15} {str(np.where(greedy_x)[0].tolist()):<20} {greedy_val:<10.3f} {greedy_val - best_val:+.3f}")
    if brute_val is not None:
        print(f"{'Brute-force':<15} {str(np.where(brute_x)[0].tolist()):<20} {brute_val:<10.3f} {brute_val - best_val:+.3f}")
    print(f"{'QAOA':<15} {str(selected_workers.tolist()):<20} {qaoa_val:<10.3f} {qaoa_val - best_val:+.3f}")
    
    # Plot optimization convergence
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(qaoa_result['optimization_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('QAOA Optimization Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    counts = qaoa_result['counts']
    top_10 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:40]
    bitstrings = [b for b, _ in top_10]
    frequencies = [c for _, c in top_10]
    best_bitstring = qaoa_result['best_bitstring']

    colors = ['red' if b == best_bitstring else 'blue' for b in bitstrings]

    plt.bar(range(len(bitstrings)), frequencies, color=colors)
    plt.xlabel('Top 10 Bitstrings')
    plt.ylabel('Counts')
    plt.title('Measurement Distribution')
    plt.xticks(range(len(bitstrings)), bitstrings, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'workers': workers,
        'scores': scores,
        'greedy_x': greedy_x,
        'greedy_val': greedy_val,
        'brute_x': brute_x,
        'brute_val': brute_val,
        'qaoa_x': x_qaoa,
        'qaoa_val': qaoa_val,
        'qaoa_result': qaoa_result,
        'lambda': lam
    }


# -------------------------
# Run demo 
# -------------------------

if __name__ == "__main__":
    res = demo(N=10, K=4) # Example with 10 (random) candidates, select 4
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)