# Bayesian Learning: Probabilistic Reasoning in ML (PYQ 7a - 2024, PYQ 7a - May 2023, PYQ 6b - 2022, PYQ 7b - 2022, PYQ 8a(iii) - CBGS)

## 1. Introduction to Bayesian Learning

**Bayesian Learning** is a probabilistic approach to machine learning based on **Bayes' Theorem**. It provides a framework for reasoning about uncertainty and updating beliefs about hypotheses as more evidence (data) becomes available.

**Core Idea:** Instead of finding a single "best" hypothesis, Bayesian methods often aim to determine the probability of each hypothesis being true, given the observed data. This allows for a more nuanced understanding of model uncertainty and can lead to more robust predictions.

**Key Characteristics:**
*   **Probabilistic:** Deals with probabilities of hypotheses and data.
*   **Incorporates Prior Knowledge:** Allows existing beliefs (priors) to be combined with new data.
*   **Updates Beliefs:** As more data is observed, beliefs (posterior probabilities) are updated.

## 2. Bayes' Theorem: The Foundation

Bayes' Theorem describes how to update the probability of a hypothesis `H` based on new evidence `E`.

**The Formula:**

`P(H|E) = [P(E|H) * P(H)] / P(E)`

Let's break down each term:

*   **`P(H|E)`: Posterior Probability**
    *   This is the probability of the hypothesis `H` being true *after* observing the evidence `E`.
    *   It represents our updated belief in the hypothesis.
    *   **Example:** The probability that a patient has a specific disease (`H`) given that their test result was positive (`E`).

*   **`P(E|H)`: Likelihood**
    *   This is the probability of observing the evidence `E` *if* the hypothesis `H` were true.
    *   It measures how well the hypothesis `H` explains the observed evidence.
    *   **Example:** The probability that a patient would test positive (`E`) if they actually have the disease (`H`). (This is related to the sensitivity of the test).

*   **`P(H)`: Prior Probability**
    *   This is the probability of the hypothesis `H` being true *before* observing any evidence `E`.
    *   It represents our initial belief or knowledge about the hypothesis.
    *   **Example:** The general prevalence of the disease in the population (`H`) before any specific test results.

*   **`P(E)`: Evidence Probability (or Marginal Likelihood)**
    *   This is the overall probability of observing the evidence `E`, regardless of whether the hypothesis `H` is true or not.
    *   It acts as a normalization constant, ensuring that the posterior probabilities sum to 1 over all possible hypotheses.
    *   It can be calculated by summing over all possible hypotheses `H_i`:
        `P(E) = Σ_i P(E|H_i) * P(H_i)`
    *   **Example:** The overall probability of a patient testing positive (`E`), considering both those who have the disease and those who don't but might get a false positive.

**How Bayes' Theorem is Used in Learning:**
In machine learning, hypotheses often correspond to different models or parameter settings. As we collect data (evidence), we use Bayes' Theorem to update the probabilities of these hypotheses. Hypotheses that better explain the data (higher likelihood `P(E|H)`) and/or have higher prior probabilities `P(H)` will have higher posterior probabilities `P(H|E)`.

**Simple Example: Medical Diagnosis**
Suppose:
*   A disease (`D`) has a prevalence of 1% in the population: `P(D) = 0.01` (Prior for having the disease).
    *   Therefore, `P(Not D) = 0.99` (Prior for not having the disease).
*   A test for this disease has the following properties:
    *   Sensitivity: If a person has the disease, the test is positive 90% of the time: `P(Positive Test | D) = 0.90` (Likelihood of positive test given disease).
    *   Specificity: If a person does not have the disease, the test is negative 95% of the time. This means the false positive rate is 5%: `P(Positive Test | Not D) = 0.05` (Likelihood of positive test given no disease).

**Question:** If a person tests positive, what is the probability they actually have the disease? We want to find `P(D | Positive Test)`.

1.  **Identify Terms:**
    *   `H = D` (Hypothesis: person has the disease)
    *   `E = Positive Test` (Evidence: person tested positive)
    *   `P(D) = 0.01`
    *   `P(Positive Test | D) = 0.90`

2.  **Calculate `P(Positive Test)` (the evidence probability):**
    `P(Positive Test) = P(Positive Test | D) * P(D) + P(Positive Test | Not D) * P(Not D)`
    `P(Positive Test) = (0.90 * 0.01) + (0.05 * 0.99)`
    `P(Positive Test) = 0.009 + 0.0495 = 0.0585`

3.  **Apply Bayes' Theorem:**
    `P(D | Positive Test) = [P(Positive Test | D) * P(D)] / P(Positive Test)`
    `P(D | Positive Test) = (0.90 * 0.01) / 0.0585`
    `P(D | Positive Test) = 0.009 / 0.0585 ≈ 0.1538` (or about 15.4%)

**Interpretation:** Even with a positive test, the probability of actually having the disease is only about 15.4%. This is because the disease is rare (low prior `P(D)`), and there's a chance of false positives.

## 3. Key Concepts in Bayesian Learning

*   **Hypotheses (H):** These can be specific parameter values for a model, different model structures, or any propositions whose truth we want to determine.
*   **Prior Distribution (P(H)):** Represents our beliefs about the hypotheses *before* observing any data. Choosing appropriate priors can be important and sometimes challenging.
*   **Likelihood (P(E|H)):** Quantifies how probable the observed data `E` is, given a particular hypothesis `H`. This is typically defined by the model we choose (e.g., a linear regression model, a Gaussian model).
*   **Evidence (P(E) or Marginal Likelihood):** The probability of the data `E` integrated over all possible hypotheses. `P(E) = ∫ P(E|H)P(H) dH` for continuous hypotheses or `Σ P(E|H_i)P(H_i)` for discrete hypotheses. It normalizes the posterior.
*   **Posterior Distribution (P(H|E)):** Represents our updated beliefs about the hypotheses *after* observing the data. This is the central quantity of interest in Bayesian inference.

*   **Maximum A Posteriori (MAP) Hypothesis:**
    A common approach is to choose the hypothesis `H_MAP` that maximizes the posterior probability:
    `H_MAP = argmax_H P(H|E) = argmax_H [P(E|H) * P(H)] / P(E)`
    Since `P(E)` is constant for all hypotheses, this simplifies to:
    `H_MAP = argmax_H [P(E|H) * P(H)]`
    MAP estimation finds a balance between the likelihood of the data given the hypothesis and the prior belief in the hypothesis.

*   **Maximum Likelihood Estimation (MLE) Hypothesis:**
    If we assume a uniform prior `P(H)` (i.e., all hypotheses are equally likely beforehand), then MAP estimation reduces to MLE:
    `H_MLE = argmax_H P(E|H)`
    MLE chooses the hypothesis that makes the observed data most probable, without considering prior beliefs.

## 4. Bayesian Networks (Probabilistic Graphical Models)

**Definition:** A Bayesian Network (also known as a Belief Network or Probabilistic Directed Acyclic Graphical Model) is a way to represent probabilistic relationships among a set of random variables.

*   It's a **Directed Acyclic Graph (DAG)**:
    *   **Nodes:** Represent random variables (e.g., `Temperature`, `Rain`, `TrafficJam`).
    *   **Edges (Arrows):** Represent direct probabilistic dependencies or causal influences. An arrow from node X to node Y means X has a direct influence on Y.
*   It's **Probabilistic:** Each node is associated with a probability distribution conditional on its parents.

**Purpose:**
*   To provide a compact and intuitive representation of a joint probability distribution over all variables in the network.
*   To perform probabilistic inference, i.e., to reason about the probabilities of certain events given evidence about other events.

**Key Components:**
1.  **Graph Structure (DAG):** Defines the qualitative relationships (dependencies) between variables.
2.  **Parameters (Conditional Probability Tables - CPTs):** For each node, a CPT quantifies the probability distribution of that node given the values of its parent nodes. `P(Node | Parents(Node))`. If a node has no parents, its CPT is just its prior probability `P(Node)`.

**The Chain Rule for Bayesian Networks:**
The joint probability distribution of all variables `X_1, ..., X_n` in a Bayesian Network can be factored as the product of the conditional probabilities of each node given its parents:
`P(X_1, ..., X_n) = Π_{i=1}^{n} P(X_i | Parents(X_i))`
This factorization significantly reduces the number of parameters needed to specify the joint distribution compared to a full joint probability table.

**Example: Simple Bayesian Network**
Consider a scenario with three variables: `Rain (R)`, `Sprinkler (S)`, and `GrassWet (W)`.
*   `Rain` can influence whether the `GrassWet`.
*   `Sprinkler` can influence whether the `GrassWet`.
*   `Rain` might influence the decision to turn the `Sprinkler` on (e.g., if it's raining, the sprinkler is less likely to be on). (For simplicity, let's initially assume Rain and Sprinkler are independent for this CPT example, but a more complex model could link them).

**Structure:**
`Rain (R)   Sprinkler (S)`
`  \       /`
`   \     /`
`    V   V`
`  GrassWet (W)`

**(Simplified) CPTs (assuming binary variables True/False):**
*   `P(R=True) = 0.2` (Prior for Rain)
*   `P(S=True) = 0.1` (Prior for Sprinkler, assuming independence from Rain for this example)
*   `P(W=True | R=True,  S=True)  = 0.99`
*   `P(W=True | R=True,  S=False) = 0.90`
*   `P(W=True | R=False, S=True)  = 0.90`
*   `P(W=True | R=False, S=False) = 0.01`

**Inference in Bayesian Networks:**
Once the network structure and CPTs are defined, we can perform inference. For example:
*   **Predictive Inference:** If we observe `Rain=True`, what is `P(GrassWet=True)`?
*   **Diagnostic Inference:** If we observe `GrassWet=True`, what is `P(Rain=True)`?
These calculations can be complex for large networks but are the core utility of Bayesian Networks.

**Learning Bayesian Networks:**
*   **Structure Learning:** Determining the graph structure (which nodes are connected) from data. This is a hard problem.
*   **Parameter Learning:** Estimating the CPTs from data, given a fixed graph structure. This is more straightforward (e.g., using frequency counts or Bayesian estimation).

## 5. Advantages of Bayesian Learning

*   **Handles Uncertainty:** Explicitly models uncertainty using probabilities.
*   **Incorporates Prior Knowledge:** Priors allow integration of domain expertise or previous findings.
*   **Provides Probabilistic Predictions:** Outputs are often probability distributions, giving a measure of confidence.
*   **Avoids Overfitting (with proper priors):** Priors can regularize models, preventing them from fitting noise in the data too closely.
*   **Bayesian Networks:** Offer an intuitive way to model complex dependencies and perform causal reasoning.
*   **Naturally handles missing data.**

## 6. Disadvantages of Bayesian Learning

*   **Computational Cost:** Inference can be computationally intensive (NP-hard in general for Bayesian Networks). Approximations like MCMC (Markov Chain Monte Carlo) are often needed.
*   **Choice of Priors:** The selection of priors can be subjective and can significantly influence the results, especially with limited data. Non-informative priors are sometimes used, but their choice is not always trivial.
*   **Model Complexity:** Defining complex Bayesian models can be challenging.

## 7. Common Applications

*   **Spam Filtering:** Naive Bayes classifiers are widely used for classifying emails as spam or not spam.
*   **Medical Diagnosis:** Building diagnostic systems based on symptoms and test results (as in the example).
*   **Document Classification:** Categorizing text documents.
*   **Risk Assessment:** In finance and insurance.
*   **Bioinformatics:** Gene sequence analysis, protein structure prediction.
*   **Recommendation Systems.**

## 8. Summary for Exams

*   **Bayesian Learning:** A probabilistic approach using **Bayes' Theorem** to update beliefs.
*   **Bayes' Theorem Formula:** `P(H|E) = [P(E|H) * P(H)] / P(E)`
    *   `P(H|E)`: **Posterior** (updated belief)
    *   `P(E|H)`: **Likelihood** (data probability given hypothesis)
    *   `P(H)`: **Prior** (initial belief)
    *   `P(E)`: **Evidence** (normalization)
*   **MAP Hypothesis:** `argmax_H [P(E|H) * P(H)]` (most probable hypothesis given data and prior).
*   **MLE Hypothesis:** `argmax_H P(E|H)` (assumes uniform prior; hypothesis that best explains data).
*   **Bayesian Networks:**
    *   **DAGs** representing probabilistic relationships between variables (nodes=variables, edges=dependencies).
    *   Uses **CPTs** (`P(Node | Parents)`) to quantify relationships.
    *   Factorizes joint probability: `P(X_1, ..., X_n) = Π P(X_i | Parents(X_i))`.
    *   Used for **probabilistic inference**.
*   **Key Idea:** Combine prior knowledge with observed data to make probabilistic conclusions.

Understanding Bayes' Theorem and the basic structure/purpose of Bayesian Networks is crucial. 