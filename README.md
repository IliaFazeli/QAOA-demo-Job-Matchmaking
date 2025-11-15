# QAOA demo: Job Matching (select K out of N workers for a shift)

This is a small experiment to explore whether a QAOA (Quantum Approximate Optimization Algorithm) can be used for finding the top K most ideal candidate workers for a posted shift on an online shift marketplace. It includes:

- Dataset generator with N candidates having certain previous ratings, distance from job,and skillset correspondence to job requirements (all randomly generated)
- Classical optimization baseline for comparison (Greedy & Brute-force)
- QAOA builder (with QUBO problem) and solver (using estimator/sampler) using QAOAAnsatz

# Results

All methods work consistently below the threshold of 10 candidates - but unfortunately QAOA starts to err beyond that limit - (suspected to be due to numerical errors on behalf of the estimator). One main reason why QAOA seems to be easily outperformed by the classical methods may be that the problem is local (scores are independent), which is perfectly exploited by greedy. 

# QUBO Formulation Derivation

This section outlines how the QUBO used in the job-matching QAOA demo is constructed.  # QUBO Formulation Derivation
# QUBO Formulation Derivation

Below is a clean, structured derivation of the QUBO used in the job-matching QAOA project.  
I kept your logic and writing style — only filled in missing steps and made it readable.

---

## Step 1 — Score Model

Each worker gets a real-valued score:

$ Score_i = \alpha \cdot skill_i + \beta \cdot rating_i - \gamma \cdot distance_i $

For this project:

$ \alpha = 1, \; \beta = 1, \; \gamma = 0.5 $

---

## Step 2 — Penalty $\lambda$

To enforce selecting exactly $K$ workers, I use:

$ \lambda = \max(Score) \cdot 2K $

This makes constraint violations more expensive than any possible gain.

---

## Step 3 — QUBO Objective

We want to minimize:

$ -\sum_i Score_i x_i + \lambda \, (\sum_i x_i - K)^2 $

where $x_i \in \{0,1\}$ represents selecting worker $i$.

### Expand the penalty term

$ (\sum_i x_i - K)^2 = (\sum_i x_i)^2 - 2K \sum_i x_i + K^2 $

and

$ (\sum_i x_i)^2 = \sum_i x_i + 2\sum_{i<j} x_i x_j $

Putting this together:

$ \lambda \sum_i x_i + 2\lambda \sum_{i<j} x_i x_j - 2\lambda K \sum_i x_i + \lambda K^2 $

---

## Combine all terms

Linear terms:

$ \sum_i (-Score_i + \lambda(1 - 2K))\, x_i $

Quadratic terms:

$ 2\lambda \sum_{i<j} x_i x_j $

Constant term:

$ C = \lambda K^2 $

---

## Step 4 — Convert QUBO to Ising Form

Use the standard substitution:

$ x_i = (1 - Z_i)/2 $

This converts binary variables into Pauli-Z operators ($Z_i = \pm 1$).

---

## Step 5 — Final Ising Hamiltonian

The Hamiltonian becomes:

$ H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + C $

with:

$ h_i = \frac{\lambda(1 - 2K) - Score_i}{2} $

$ J_{ij} = \frac{\lambda}{2} $

$ C = \lambda K^2 $

This is exactly what is implemented in the SparsePauliOp used in the QAOA solver.

---

## Summary
