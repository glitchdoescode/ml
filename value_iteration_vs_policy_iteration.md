# Value Iteration vs. Policy Iteration: Finding Optimal Policies (PYQ 6b - 2024, PYQ 7b - CBGS)

## 1. Introduction

Value Iteration (VI) and Policy Iteration (PI) are two fundamental dynamic programming algorithms used in Reinforcement Learning (RL) to find the optimal policy (`π*`) for a given Markov Decision Process (MDP). Both are **model-based** algorithms, meaning they require full knowledge of the MDP's dynamics: the state transition probabilities `P(s'|s,a)` and the reward function `R(s,a,s')`.

Their goal is to compute the optimal state-value function (`V*(s)`) or the optimal action-value function (`Q*(s,a)`), from which the optimal policy can be derived.

## 2. Recap: Key Concepts

*   **State-Value Function (`V(s)`):** The expected cumulative discounted reward starting from state `s` and following a specific policy `π` (`V^π(s)`) or the optimal policy (`V*(s)`).
*   **Policy (`π(s)`):** A mapping from states to actions, defining the agent's behavior.
*   **Bellman Equations:** Provide the recursive relationships defining value functions. Value Iteration and Policy Iteration are essentially algorithms for solving these equations.
    *   **Bellman Optimality Equation for V*(s):** `V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]`
    *   **Bellman Equation for V^π(s):** `V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]`

## 3. Value Iteration (VI)

**Goal:** To directly find the optimal state-value function `V*(s)`.

**Core Idea:** Value Iteration repeatedly applies the Bellman Optimality Equation as an update rule for `V(s)`. It starts with an arbitrary guess for `V(s)` and iteratively refines this estimate until it converges to `V*(s)`.

**Algorithm Steps:**
1.  **Initialization:** For all states `s ∈ S`, initialize `V_0(s)` to arbitrary values (e.g., 0).
    Set `k = 0` (iteration counter).
2.  **Iteration Loop:** Repeat until `V_k(s)` converges (e.g., the maximum change `max_s |V_{k+1}(s) - V_k(s)|` is below a small threshold `θ`):
    a.  Increment `k ← k + 1`.
    b.  For each state `s ∈ S`:
        `V_k(s) ← max_a Σ_{s'∈S} P(s'|s,a) [R(s,a,s') + γV_{k-1}(s')]`
        (This step updates the value of state `s` based on the values of successor states from the *previous iteration* `V_{k-1}`. It finds the action `a` that maximizes the expected one-step lookahead reward plus the discounted value of the next state.)
3.  **Output (Optimal Policy Extraction):** Once `V_k(s)` has converged to `V*(s)`, the optimal deterministic policy `π*(s)` can be extracted by choosing the action that maximizes the expected utility for each state:
    `π*(s) = argmax_a Σ_{s'∈S} P(s'|s,a) [R(s,a,s') + γV*(s')]`

**Formula Explanation (Update Rule):**
*   `V_k(s)`: Value of state `s` at iteration `k`.
*   `max_a`: Take the maximum over all possible actions `a` available in state `s`.
*   `Σ_{s'∈S} P(s'|s,a) [...]`: The sum over all possible next states `s'`, weighted by the probability of transitioning to `s'` given state `s` and action `a`.
*   `R(s,a,s')`: Immediate reward for transitioning from `s` to `s'` via action `a`.
*   `γV_{k-1}(s')`: Discounted value of the next state `s'`, using the value function from the previous iteration `k-1`.

**Convergence:** Value Iteration is guaranteed to converge to the optimal state-value function `V*(s)` as `k → ∞` because the Bellman operator is a contraction mapping.

## 4. Policy Iteration (PI)

**Goal:** To find the optimal policy `π*`.

**Core Idea:** Policy Iteration alternates between two main steps:
1.  **Policy Evaluation:** Given the current policy `π`, calculate the state-value function `V^π(s)` for that policy.
2.  **Policy Improvement:** Improve the current policy `π` by acting greedily with respect to `V^π(s)` to create a new policy `π'`.
These two steps are repeated until the policy no longer changes, at which point it is guaranteed to be optimal.

**Algorithm Steps:**
1.  **Initialization:**
    a.  Initialize a policy `π_0(s)` arbitrarily for all `s ∈ S` (e.g., a random policy).
    b.  Initialize `V(s)` arbitrarily (e.g., to 0), or it will be computed in the first evaluation step.
    Set `k = 0`.
2.  **Iteration Loop:** Repeat until `π_k(s)` converges (i.e., `π_{k+1}(s) = π_k(s)` for all `s`):
    a.  **Step 1: Policy Evaluation (Compute `V^{π_k}(s)` for current policy `π_k`)**
        *   For the current policy `π_k`, calculate its value function `V^{π_k}(s)`. This involves solving the system of linear equations defined by the Bellman Equation for `V^{π_k}`:
            `V^{π_k}(s) = Σ_{a} π_k(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^{π_k}(s')]`
        *   In practice, this is often done iteratively for a fixed number of steps or until `V^{π_k}(s)` converges for the current `π_k`:
            Initialize `V(s)` (e.g., to 0 or `V^{π_{k-1}}(s)`).
            Loop until `V(s)` converges (for current `π_k`):
                For each state `s ∈ S`:
                    `V_{new}(s) ← Σ_a π_k(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV_{old}(s')]`
            Set `V^{π_k}(s) ← V_{new}(s)`.

    b.  **Step 2: Policy Improvement (Generate `π_{k+1}(s)` based on `V^{π_k}(s)`)**
        *   For each state `s ∈ S`, improve the policy by choosing the action that maximizes the expected utility according to `V^{π_k}(s)`:
            `π_{k+1}(s) ← argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^{π_k}(s')]`
        *   This new policy `π_{k+1}` is greedy with respect to `V^{π_k}`.

    c.  **Check for Convergence:** If `π_{k+1}(s) = π_k(s)` for all `s`, then stop; `π_{k+1}` is the optimal policy `π*` and `V^{π_k}` is `V*`.
        Otherwise, increment `k ← k + 1` and go back to step 2a.

**Convergence:** Policy Iteration is guaranteed to converge to the optimal policy `π*` in a finite number of iterations (for finite MDPs) because there are only a finite number of policies, and each policy improvement step strictly improves the policy unless it's already optimal.

## 5. Value Iteration vs. Policy Iteration: Head-to-Head

| Feature                     | Value Iteration                                     | Policy Iteration                                                                 |
| --------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Primary Goal**            | Find optimal value function `V*(s)` first.           | Find optimal policy `π*(s)` directly.                                           |
| **Main Operation**          | Iteratively apply Bellman Optimality update.       | Alternates Policy Evaluation and Policy Improvement.                           |
| **Policy Evaluation Step**  | Not explicitly separate; embedded in `max_a`.     | Explicit step: solve for `V^π(s)` for the current `π` (can be iterative itself). |
| **Policy Improvement Step** | Done once at the end after `V*(s)` converges.      | Done in each iteration after Policy Evaluation.                                  |
| **Cost per Iteration**      | One pass through all states, updating `V(s)` using `max_a`. Relatively cheaper. | Policy Evaluation can be computationally expensive (multiple sweeps through states). Policy Improvement is similar to one VI sweep. | 
| **Number of Iterations**    | May require many iterations for `V(s)` to converge. | Often converges in fewer iterations (policy space is typically smaller than value space). |
| **Overall Complexity**      | Can be `O(|A||S|^2)` per iteration if transitions are dense; total complexity depends on convergence rate. | Policy Evaluation can be `O(|S|^3)` if solved directly, or multiple sweeps if iterative. Often faster if Policy Evaluation converges quickly. | 
| **When `V(s)` is updated**  | `V(s)` values are updated continuously towards `V*(s)`. | `V(s)` is computed for a fixed policy `π` in evaluation, then `π` is updated.      |
| **Termination**             | When `V(s)` values stabilize.                       | When the policy `π(s)` stabilizes.                                               |

## 6. When to Use Which?

*   **Value Iteration:**
    *   Generally simpler to implement.
    *   Can be more efficient if the number of actions is small or if one full policy evaluation step in PI is very costly.
    *   Often preferred when the primary goal is to get the optimal values `V*(s)` quickly, even if the policy is extracted later.

*   **Policy Iteration:**
    *   Often converges in fewer iterations than Value Iteration, especially if the policy space is small or if good initial policies are known.
    *   Can be more efficient if the policy evaluation step can be done very quickly (e.g., if only a few iterations are needed for `V^π` to converge, or if states are few).
    *   If the number of actions `|A|` is very large, PI might be better as VI's `max_a` becomes expensive in each step for all states.

In many practical scenarios, especially with a small number of states, Policy Iteration converges faster in terms of wall-clock time if its evaluation step is efficient. However, if the state space is very large, the full policy evaluation sweep in PI can be a bottleneck.

**Modified Policy Iteration:** Sometimes, the Policy Evaluation step in PI is not run to full convergence but only for a fixed number of iterations. This can speed up PI.

## 7. Simple Grid World Example

Imagine a simple grid where the agent wants to reach a goal state. Both algorithms would start with initial estimates.

*   **Value Iteration:** Would iteratively update the value of each cell based on the max expected value it could get by moving to neighboring cells, until the values across the grid stabilize. Then it would determine the best direction to move from each cell.
*   **Policy Iteration:** Might start with a random policy (e.g., 25% chance of moving N, S, E, W from each cell). 
    1.  **Evaluate:** Calculate how good that random policy is (the `V^π` for each cell).
    2.  **Improve:** For each cell, see if there's a better direction to move given the `V^π` values. Update the policy.
    Repeat until the suggested directions (policy) don't change.

Both will eventually find the optimal path, but the intermediate steps and computational focus differ.

## 8. Advantages and Disadvantages

**Value Iteration:**
*   **Advantages:** Simpler to implement; each iteration is computationally less expensive than a full PI iteration.
*   **Disadvantages:** May require many iterations to converge; only finds the optimal policy at the very end.

**Policy Iteration:**
*   **Advantages:** Often converges in fewer iterations; always maintains a valid policy throughout the process.
*   **Disadvantages:** Each iteration (especially policy evaluation) can be computationally expensive; slightly more complex to implement.

## 9. Summary for Exams (PYQ 6b - 2024, PYQ 7b - CBGS)

| Aspect             | Value Iteration (VI)                                  | Policy Iteration (PI)                                               |
|--------------------|-------------------------------------------------------|---------------------------------------------------------------------|
| **Goal**           | Find `V*(s)` then `π*(s)`                             | Iteratively find `π*(s)` (evaluates `V^π(s)` for current `π`)           |
| **Process**        | `V_k(s) ← BellmanOptimalityUpdate(V_{k-1})`             | `π → Evaluate V^π → Improve π' → π`                                |
| **Policy Eval.**   | Implicit in `max_a`                                   | Explicit step, can be iterative                                     |
| **Complexity/Iter**| Lower                                                 | Higher (due to Policy Evaluation)                                   |
| **Num. Iters**     | Potentially many                                      | Often fewer                                                         |
| **Guaranteed Opt?**| Yes                                                   | Yes                                                                 |

*   Both are **model-based** (need `P` and `R`).
*   Both solve Bellman equations to find the optimal policy in an MDP.
*   **Value Iteration** focuses on updating the state-value function towards the optimal one.
*   **Policy Iteration** alternates between evaluating the current policy and improving it greedily.
*   The choice between them can depend on the specific characteristics of the MDP (number of states, actions, cost of evaluation).

Both methods are foundational for understanding more advanced RL algorithms. 