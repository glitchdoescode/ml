# Markov Decision Process (MDP): Framework for Reinforcement Learning (PYQ 8iv - May 2024, PYQ 5b - 2022, PYQ 7b - CBGS)

## 1. What is a Markov Decision Process (MDP)?

A **Markov Decision Process (MDP)** is a mathematical framework used to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker (an **agent**). It provides a formal way to describe an **environment** in Reinforcement Learning (RL) where the outcomes are not entirely predictable.

**Key Idea:** An agent interacts with an environment over a sequence of discrete time steps. At each step:
1.  The agent observes the environment's current **state**.
2.  The agent chooses an **action** based on this state.
3.  The environment transitions to a new **state** based on the current state and the chosen action.
4.  The agent receives a numerical **reward** (or penalty) from the environment.

The term **"Markov"** refers to the **Markov property**: the future state and reward depend *only* on the current state and action, not on the sequence of states and actions that preceded it (i.e., the past is irrelevant given the present).

**Purpose in Reinforcement Learning:**
MDPs are fundamental to RL because they provide the formal specification of the interaction between the agent and the environment. The goal of an RL agent operating within an MDP is to learn a **policy** (a strategy for choosing actions) that maximizes the cumulative reward over time.

## 2. Components of an MDP

An MDP is typically defined by a tuple `(S, A, P, R, γ)`:

### a) States (S)
*   **Definition:** A finite set of all possible situations or configurations the agent can be in.
*   **Role:** Represents the information the agent has about the environment at a given time.
*   **Example:** In a grid world, each cell can be a state. In a chess game, the specific arrangement of pieces on the board is a state. For a robot, its position and sensor readings could define the state.
*   `s ∈ S` denotes a specific state. `S_t` is the state at time step `t`.

### b) Actions (A)
*   **Definition:** A finite set of all possible moves or choices the agent can make.
*   **Role:** These are the means by which the agent interacts with and influences the environment.
*   **Example:** In a grid world, actions could be `up, down, left, right`. In chess, actions are all valid moves for the pieces. For a robot, actions could be "move forward," "turn left," "pick up object."
*   `a ∈ A` denotes a specific action. `A_t` is the action taken at time step `t`.
*   Sometimes actions can be state-dependent, denoted `A(s)` (the set of actions available in state `s`).

### c) Transition Probability Function (P or T)
*   **Definition:** Defines the dynamics of the environment. It specifies the probability of transitioning from state `s` to a new state `s'` if the agent takes action `a`.
*   **Notation:** `P(s' | s, a) = P(S_{t+1} = s' | S_t = s, A_t = a)`
*   **Role:** Captures the (potentially stochastic/random) outcomes of the agent's actions.
*   **Example:** In a grid world, if an agent in cell (x,y) chooses to move `right`:
    *   There might be an 0.8 probability of successfully moving to (x+1,y) (the intended state `s'`).
    *   There might be a 0.1 probability of slipping and staying in (x,y) (another possible `s'`).
    *   There might be a 0.1 probability of slipping and moving to (x,y-1) (another possible `s'`).
*   The sum of probabilities over all possible next states `s'` must be 1 for any given `s` and `a`: `Σ_{s'∈S} P(s' | s, a) = 1`.

### d) Reward Function (R)
*   **Definition:** Defines the immediate feedback (a scalar value) the agent receives from the environment after transitioning from state `s` to state `s'` as a result of action `a`. It can also be defined as `R(s,a)` (reward for taking action `a` in state `s`) or `R(s)` (reward for being in state `s`).
*   **Notation:** `R(s, a, s')` is the reward for transitioning from `s` to `s'` via `a`. Often, this is the expected immediate reward `E[R_{t+1} | S_t=s, A_t=a, S_{t+1}=s']`.
*   **Role:** Guides the agent's learning process. The agent aims to maximize the cumulative sum of these rewards.
*   **Example:** In a grid world:
    *   Reaching a goal state: +10 reward.
    *   Falling into a pit state: -100 reward.
    *   Any other movement (non-terminal state): -1 reward (to encourage efficiency and reaching the goal faster).

### e) Discount Factor (γ - Gamma)
*   **Definition:** A scalar value between 0 and 1 (i.e., `0 ≤ γ ≤ 1`).
*   **Purpose:** It determines the present value of future rewards. A reward received `k` time steps in the future is worth `γ^k` times what it would be worth if received immediately.
    *   If `γ = 0`: The agent is "myopic" and only cares about the immediate reward (`R_{t+1}`).
    *   If `γ` is close to 1: The agent is "far-sighted" and values future rewards almost as much as immediate ones.
    *   If `γ = 1`: All future rewards are considered equally (can be problematic for tasks that don't have a clear end, as total reward might be infinite).
*   **Why use a discount factor?**
    1.  **Mathematical Convenience:** Prevents infinite sums of rewards in ongoing (non-terminating) tasks.
    2.  **Uncertainty:** Future rewards are often less certain than immediate ones; discounting reflects this.
    3.  **Preference Modeling:** Often reflects a preference for immediate gratification (common in biological systems).
*   **Example:** If `γ = 0.9`, a reward of 10 received one time step later is valued at `0.9 * 10 = 9` now. A reward of 10 received two time steps later is valued at `0.9^2 * 10 = 8.1` now.

## 3. The Markov Property

The "Markov" in MDP signifies that the environment satisfies the **Markov Property**. This property states that **the future is independent of the past given the present**.

More formally:
*   The probability of transitioning to state `S_{t+1}` and receiving reward `R_{t+1}` depends *only* on the current state `S_t` and the current action `A_t`.
*   `P(S_{t+1}, R_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1}, R_{t+1} | S_t, A_t)`

**Implication:** The current state `S_t` encapsulates all relevant information from the history needed to make an optimal decision. The agent doesn't need to remember the entire sequence of past states and actions to predict the future or choose the best next action.

## 4. The Goal in an MDP: Finding an Optimal Policy

The ultimate objective of an agent in an MDP is to find an **optimal policy (π*)**.

*   **Policy (π):** A mapping from states to actions, specifying which action the agent will take when in a particular state.
    *   **Deterministic Policy:** `π(s) = a` (In state `s`, always take action `a`).
    *   **Stochastic Policy:** `π(a|s) = P(A_t = a | S_t = s)` (In state `s`, take action `a` with a certain probability).

*   **Return (G_t):** The total discounted reward from time step `t` onwards.
    `G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ... = Σ_{k=0}^{∞} γ^k R_{t+k+1}`

*   **Optimal Policy (π*):** A policy `π*` is optimal if its expected return is greater than or equal to the expected return of any other policy `π'` for all states `s`.
    More formally, `π*` maximizes the expected value function `V^π(s)` for all `s ∈ S`.

Reinforcement learning algorithms (like Value Iteration, Policy Iteration, and Q-learning) are designed to find or approximate such an optimal policy, often by first finding the optimal value functions (`V*(s)` or `Q*(s,a)`).

## 5. Example: A Simple Grid World

Consider a 2x2 grid world:
`[ S0 (Start) | S1         ]`
`--------------------------`
`[ S2 (Pit)   | S3 (Goal)  ]`

*   **States (S):** `{S0, S1, S2, S3}`.
*   **Actions (A):** `{Up, Down, Left, Right}`. (Assume if an action would hit a wall, the agent stays in its current state).
*   **Transition Probabilities (P):** Let's assume actions are mostly deterministic but with a small chance of error (e.g., 0.8 probability of intended move, 0.1 probability of moving right of intended, 0.1 of moving left of intended).
    *   Example: From `S0`, action `Right`:
        *   `P(S1 | S0, Right) = 0.8` (goes to S1)
        *   `P(S0 | S0, Right) = 0.1` (slips, stays in S0 if Up/Down was alternative)
        *   `P(S2 | S0, Right) = 0.1` (slips, goes to S2 if Down was alternative)
*   **Reward Function (R):**
    *   Transitioning to `S3` (Goal): +10
    *   Transitioning to `S2` (Pit): -10
    *   Any other transition: -1 (cost for each step)
*   **Discount Factor (γ):** e.g., `0.9`.

**Markov Property in this Example:** If the agent is in `S0`, the probabilities of reaching `S1`, `S0`, or `S2` after taking action `Right`, and the associated rewards, depend *only* on being in `S0` and choosing `Right`. It doesn't matter how the agent arrived at `S0` (e.g., from `S1` with action `Left`, or if it started there).

**Goal:** The agent wants to learn a policy (e.g., from `S0`, should it go `Up, Down, Left, or Right`?) that maximizes its sum of discounted rewards, helping it reach `S3` while avoiding `S2` and minimizing steps.

## 6. Summary for Exams (PYQ 8iv - May 2024, PYQ 5b - 2022, PYQ 7b - CBGS)

*   **MDP Definition:** A mathematical framework for modeling sequential decision-making under uncertainty, forming the basis for most reinforcement learning problems.
*   **Key Components (Tuple `(S, A, P, R, γ)`):**
    *   **S (States):** Set of all possible situations.
    *   **A (Actions):** Set of choices available to the agent.
    *   **P (Transition Probability `P(s'|s,a)`):** Defines environment dynamics; probability of next state given current state and action.
    *   **R (Reward Function `R(s,a,s')`):** Immediate feedback signal from the environment.
    *   **γ (Discount Factor):** Scalar (0 to 1) that trades off immediate vs. future rewards.
*   **Markov Property:** The future state and reward depend only on the current state and action, not on the past history.
*   **Agent's Goal:** To find an optimal **policy (π*)** – a rule for choosing actions in states – that maximizes the **expected cumulative discounted reward** (Return `G_t`).

MDPs provide the essential language and structure for defining and solving reinforcement learning problems. 