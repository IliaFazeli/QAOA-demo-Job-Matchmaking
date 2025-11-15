# QAOA demo: Job Matching (select K out of N workers for a shift)

This is a small experiment to explore whether a QAOA (Quantum Approximate Optimization Algorithm) can be used for finding the top K most ideal candidate workers for a posted shift on an online shift marketplace. It includes:

- Dataset generator with N candidates having certain previous ratings, distance from job,and skillset correspondence to job requirements (all randomly generated)
- Classical optimization baseline for comparison (Greedy & Brute-force)
- QAOA builder (with QUBO problem) and solver (using estimator/sampler) using QAOAAnsatz

# Results

All methods work consistently below the threshold of 10 candidates - but unfortunately QAOA starts to err beyond that limit - (suspected to be due to numerical errors on behalf of the estimator). One main reason why QAOA seems to be easily outperformed by the classical methods may be that the problem is local (scores are independent), which is perfectly exploited by greedy. 

# QUBO Formulation Derivation

This section outlines how the QUBO used in the job-matching QAOA demo is constructed.  # QUBO Formulation Derivation

---
# Worker Selection QUBO → Ising

## Step 1 — Compute Worker Scores

Each worker is assigned a real-valued score:

$$
\text{Score}_i = \alpha \cdot \text{skill}_i + \beta \cdot \text{rating}_i - \gamma \cdot \text{distance}_i
$$

For this project:

$$
\alpha = 1, \quad \beta = 1, \quad \gamma = 0.5
$$

---

## Step 2 — Penalty Parameter $\lambda$

To enforce selecting **exactly $K$ workers**, define a penalty:

$$
\lambda = 2 K \cdot \max(\text{Score})
$$

This ensures that constraint violations are more expensive than any possible score gain.

---

## Step 3 — QUBO Objective

We want to **minimize**:

$$
-\sum_i \text{Score}_i x_i + \lambda \left( \sum_i x_i - K \right)^2
$$

where $x_i \in \{0,1\}$ indicates whether worker $i$ is selected.

### Expand the penalty term

$$
\left( \sum_i x_i - K \right)^2 = \left( \sum_i x_i \right)^2 - 2 K \sum_i x_i + K^2
$$

and

$$
\left( \sum_i x_i \right)^2 = \sum_i x_i + 2 \sum_{i<j} x_i x_j
$$

So the penalty becomes:

$$
\lambda \sum_i x_i + 2\lambda \sum_{i<j} x_i x_j - 2 \lambda K \sum_i x_i + \lambda K^2
$$

---

## Step 4 — Combine Terms

**Linear terms:**

$$
\sum_i (-\text{Score}_i + \lambda(1 - 2K)) x_i
$$

**Quadratic terms:**

$$
2 \lambda \sum_{i<j} x_i x_j
$$

**Constant term:**

$$
C = \lambda K^2
$$

---

## Step 5 — Convert QUBO → Ising

Use the standard substitution:

$$
x_i = \frac{1 - Z_i}{2}, \quad Z_i = \pm 1
$$

---

## Step 6 — Final Ising Hamiltonian

$$
H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + C
$$

with coefficients:

$$
h_i = \frac{\lambda(1 - 2K) - \text{Score}_i}{2}, \quad
J_{ij} = \frac{\lambda}{2}, \quad
C = \lambda K^2
$$

This corresponds directly to the **SparsePauliOp** used in the QAOA solver.

---

## Quickstart

Create and activate a virtual env, install deps:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
