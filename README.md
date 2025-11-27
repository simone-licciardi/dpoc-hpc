# Dynamic Programming Challenge
**Simone Licciardi & Samuele Cipriani**  
_For typos: `slicciardi@student.ethz.ch`_  
**Date:** `November 2025`

We detail our solution report in two sections: the details of our implementation of the DPA algorithm, and the structural characteristics of the problem that we exploited. We also attached a section about **super cool but too fancy** attempts.

---

## Problem Structure

The transition matrix **P** is sparse and has structure, but **cannot be written as a block matrix or banded matrix** in a practical implementation-friendly way. Moreover, the problem has no trivial preconditioner.  
> *“Purely algebraic approaches which simply take the numerical matrix entries as input can have merit in situations where little is known about the underlying problem—they certainly have the useful property that it is usually possible to apply such a method, but the generated preconditioners can be poor.”*  
> — [A well-known review](https://people.maths.ox.ac.uk/wathen/preconditioning.pdf)

Let the state be  
$$\mathcal{X} \ni i = (i_{YV}, i_{DH}),\quad  
i_{YV} \in \mathbb{Z}^2,\ i_{DH} \in \mathbb{Z}^{2M}.$$

We exploit the fact that the transition kernel factorizes:
\[
P(i,j|u) = 
P^{(YV)}(i_{YV},j_{YV}|u)\,
P^{(DH)}(i_{DH},j_{DH})\,
P^{(coll)}(i).
\]

This uses the following structural observations:

- The transition in DH coordinates is **independent of the action** \(u\).  
- Conditional on *no collision*, the DH transition depends only on the DH components of \(i\) and \(j\).  
- The YV transition depends only on YV coordinates.  
- The collision probability depends only on the initial state \(i\).

We computed the three component matrices in a fully vectorized way and assembled \(P\) via vectorized operations.  
All \(P^{(u)}\) are stored in **CSR format** to speed up sparse matrix operations downstream.

---

## Solver

Our solver is a hybrid between **Value Iteration (VI)** and **Policy Iteration (PI)**, designed to exploit the strengths of both.

### Key observations

- **PI** is efficient only if you can solve sparse linear systems quickly, which requires strong structural properties or good preconditioners — neither is present here.
- **VI** is linear-time when sparse matrix–vector multiplication is fast — which *is* the case for our \(P\).
- **PI** converges faster if the initialization is good.
- **VI** obtains a good *policy* long before it obtains a good *value*.

### Strategy

We iterate:

1. Run a few steps of **Value Iteration** (cheap).  
2. Extract the greedy policy.  
3. Perform **one single Policy Evaluation step**.  
4. Repeat until convergence.

A crucial implementation detail:  
We **do not** run two PI steps (the usual PI termination requirement).  
Because one PI evaluation is as expensive as ~10–20 VI steps, we rely on the following heuristic termination rule:

- If the PI step has already produced the optimal value, then the next batch of VI iterations will immediately satisfy the VI tolerance condition.

Empirically, this reduces the total number of costly Policy Evaluations.

---

## Algorithm: Hybrid Value–Policy Iteration

```text
Algorithm 1: Hybrid Value-Policy Iteration
Input: C (problem constants), N_val (VI steps per PI)
Output: J*, μ*

1. Compute P^(u) for u ∈ {0,1,2} and Q ∈ ℝ³
2. Initialize J ← -50·1,  k ← 0

3. While ||J^(k) - J^(k−1)||∞ > ε:
       # Value Iteration
       J^(k+1) ← min_u { P^(u) J^(k) + Q_u 1 }
       k ← k + 1

       If k mod N_val = 0:
           # Policy Improvement
           μ(i) ← argmin_u { P^(u) J^(k) + Q_u 1 }_i

           # Policy Evaluation
           Solve (I − P^(μ)) J^(k+1) = Q_μ

4. Extract μ* ← argmin_u { P^(u) J^(k) + Q_u 1 }
5. Return (J^(k), μ*)

# dpoc-hpc
