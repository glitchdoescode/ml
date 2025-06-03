# K-Nearest Neighbors (KNN): Learning from Proximity (PYQ 6a - May 2023)

## 1. What is K-Nearest Neighbors (KNN)?

**K-Nearest Neighbors (KNN)** is a simple, intuitive, and versatile supervised machine learning algorithm that can be used for both **classification** and **regression** tasks.

*   **Non-parametric:** KNN makes no assumptions about the underlying data distribution (e.g., it doesn't assume data is Gaussian).
*   **Instance-based Learning (or Lazy Learning):** KNN is called a "lazy learner" because it doesn't explicitly build a model during a distinct training phase. Instead, it stores all the training data points. The actual "learning" or computation happens only when a prediction is requested for a new, unseen data point.

**Core Idea:** The fundamental principle of KNN is that similar things exist in close proximity. In other words, a data point is likely to belong to the same class (for classification) or have a similar value (for regression) as its nearest neighbors in the feature space.

## 2. How KNN Works (Classification)

For classification, KNN assigns a class label to a new data point based on the majority class of its `K` closest data points in the training set.

**Algorithm Steps for Classification:**
1.  **Choose the Number of Neighbors (`K`):** Select an integer `K`, which represents the number of nearest neighbors to consider. This is a crucial hyperparameter.
2.  **For a new data point (`x_new`) to classify:**
    a.  **Calculate Distances:** Compute the distance between `x_new` and every data point (`x_i`) in the training dataset. Common distance metrics include Euclidean distance, Manhattan distance, etc.
    b.  **Identify `K` Nearest Neighbors:** Find the `K` training data points that have the smallest distances to `x_new`.
    c.  **Majority Vote:** Determine the class labels of these `K` nearest neighbors. Count the occurrences of each class label among these neighbors.
    d.  **Assign Class:** Assign the class label that has the majority vote among the `K` nearest neighbors to the new data point `x_new`. If there's a tie (and K is even, or for multi-class scenarios), a tie-breaking rule might be needed (e.g., choose randomly, or use distance weighting).

**Analogy: Asking Your Closest Friends**
Imagine you move to a new neighborhood and want to decide if a particular local restaurant is good. You might ask your `K` closest neighbors (friends you've made who live nearby) about their opinion. If the majority of them say it's good, you'd probably predict it's good too. KNN works similarly with data points.

## 3. Distance Metrics

The choice of distance metric significantly impacts KNN's performance. It defines what "nearest" means.

*   **Euclidean Distance (L2 norm):** The most common metric. It's the straight-line distance between two points `p = (p1, p2, ..., pn)` and `q = (q1, q2, ..., qn)` in an n-dimensional space.
    `Distance(p, q) = √((p1-q1)² + (p2-q2)² + ... + (pn-qn)²)`

*   **Manhattan Distance (L1 norm):** The sum of the absolute differences of their Cartesian coordinates. Imagine navigating a city grid where you can only travel along horizontal or vertical streets.
    `Distance(p, q) = |p1-q1| + |p2-q2| + ... + |pn-qn|`

*   **Minkowski Distance:** A generalization of both Euclidean and Manhattan distances.
    `Distance(p, q) = (Σ_{i=1}^{n} |pi-qi|^c)^(1/c)`
    *   If `c=1`, it's Manhattan distance.
    *   If `c=2`, it's Euclidean distance.

**Importance of Feature Scaling:**
KNN is highly sensitive to the scale of features. Features with larger ranges or magnitudes can dominate the distance calculation, even if they are not more important. Therefore, it's crucial to **normalize or standardize** the features before applying KNN (e.g., scale to [0,1] or use z-score normalization).

## 4. Choosing the Value of K

The choice of `K` is critical and can significantly affect the classification results.

*   **Small `K` (e.g., K=1):**
    *   The decision boundary can be very complex and irregular, closely following the training data.
    *   The model can be very sensitive to noise and outliers in the training data (high variance).
    *   Potential for **overfitting**.

*   **Large `K` (e.g., K close to the number of training samples):**
    *   The decision boundary becomes smoother and simpler.
    *   The model is less sensitive to noise but might ignore local patterns.
    *   Computationally more expensive as more neighbors need to be considered.
    *   Potential for **underfitting** (oversimplification, high bias); it might classify all new points to the majority class of the entire dataset.

**Methods for Choosing `K`:**
*   There's no single best `K` for all datasets.
*   `K` is often chosen by **cross-validation** (e.g., trying different values of `K` and selecting the one that gives the best performance on a validation set).
*   A common rule of thumb is to use `K = √N`, where N is the number of samples in the training set.
*   For binary classification, `K` is often chosen as an **odd number** to avoid ties in the majority vote. If `K` is even and a tie occurs, one might need a tie-breaking strategy (e.g., reduce K by 1, or choose the class with the smallest average distance to its neighbors).

**Distance Weighting:** Instead of a simple majority vote, neighbors can be weighted by the inverse of their distance. Closer neighbors get a higher weight in the voting process. This can help mitigate the effect of a less optimal `K`.

## 5. KNN for Regression

KNN can also be used for regression tasks. The process is similar, but the final prediction step changes:

1.  Choose `K` and calculate distances to find the `K` nearest neighbors for a new data point.
2.  Instead of a majority vote for the class, the prediction for the new data point is typically the **average (or median)** of the target values (continuous values) of its `K` nearest neighbors.

    `ŷ_new = (1/K) * Σ_{i=1}^{K} y_i` (where `y_i` are the target values of the K neighbors)

## 6. Advantages of KNN

*   **Simple and Intuitive:** Easy to understand the underlying mechanism.
*   **Easy to Implement:** The algorithm itself is straightforward.
*   **No Explicit Training Phase (Lazy Learner):** The model is simply the stored training data. New data points can be added easily without retraining the entire model.
*   **Naturally Handles Multi-Class Problems:** The majority voting scheme extends directly to multi-class classification.
*   **Versatile:** Can be used for both classification and regression.
*   **Can form complex decision boundaries:** With a small K, it can adapt to local data structures.

## 7. Disadvantages of KNN

*   **Computationally Expensive at Prediction Time:** For each new prediction, distances to ALL training points must be computed. This can be very slow for large datasets (O(N*d) where N is number of training samples and d is number of dimensions).
    *   Specialized data structures like KD-trees or Ball trees can speed up the search for nearest neighbors, but their effectiveness diminishes in high dimensions.
*   **Requires Significant Memory:** Needs to store the entire training dataset.
*   **Sensitive to the Choice of `K`:** Performance heavily depends on `K`.
*   **Sensitive to the Choice of Distance Metric:** The right metric is problem-dependent.
*   **Curse of Dimensionality:** Performance degrades in high-dimensional spaces. As the number of dimensions increases, data points become sparse, and the concept of "nearest" or "local neighborhood" becomes less meaningful. Distances between points tend to become more uniform.
*   **Sensitive to Irrelevant Features:** Features that don't contribute to distinguishing classes can mislead the distance calculations. Feature selection or weighting can be important.
*   **Sensitive to Feature Scaling:** As mentioned, features must be scaled.
*   **Imbalanced Data:** KNN can be biased towards the majority class in imbalanced datasets because neighbors from the majority class are often more numerous.

## 8. When to Use KNN

*   For smaller datasets where computational cost at prediction is not a major concern.
*   When the decision boundary is highly irregular and non-linear.
*   As a baseline model to compare against more complex algorithms.
*   When interpretability of why a prediction is made (by looking at neighbors) is useful.
*   For applications where data is continuously updated, as new data points can be added without retraining.

## 9. Simple Example (Classification)

Consider a 2D dataset with two classes (A and B) and `K=3`.
Training Data:
*   P1: (1,2), Class A
*   P2: (2,3), Class A
*   P3: (3,1), Class A
*   P4: (6,5), Class B
*   P5: (7,7), Class B
*   P6: (8,6), Class B

New point to classify: `X_new = (3,2)`

1.  **Calculate Distances (e.g., Euclidean):**
    *   Dist(X_new, P1) = √((3-1)²+(2-2)²) = √(2²+0²) = 2
    *   Dist(X_new, P2) = √((3-2)²+(2-3)²) = √(1²+(-1)²) = √2 ≈ 1.41
    *   Dist(X_new, P3) = √((3-3)²+(2-1)²) = √(0²+1²) = 1
    *   Dist(X_new, P4) = √((3-6)²+(2-5)²) = √((-3)²+(-3)²) = √18 ≈ 4.24
    *   Dist(X_new, P5) = √((3-7)²+(2-7)²) = √((-4)²+(-5)²) = √41 ≈ 6.40
    *   Dist(X_new, P6) = √((3-8)²+(2-6)²) = √((-5)²+(-4)²) = √41 ≈ 6.40

2.  **Identify K=3 Nearest Neighbors:**
    *   P3: (3,1), Class A, Distance = 1
    *   P2: (2,3), Class A, Distance ≈ 1.41
    *   P1: (1,2), Class A, Distance = 2

3.  **Majority Vote:**
    *   Class A: 3 votes
    *   Class B: 0 votes

4.  **Assign Class:** `X_new` is classified as Class A.

## 10. Summary for Exams (PYQ 6a - May 2023)

*   **KNN:** A simple, non-parametric, instance-based (lazy) learning algorithm.
*   **Core Idea (Classification):** Classifies a new point based on the **majority class of its `K` nearest neighbors** in the training data.
*   **Core Idea (Regression):** Predicts the value of a new point based on the **average/median of the values of its `K` nearest neighbors**.
*   **Key Steps:** Choose `K`, calculate distances, find `K` nearest neighbors, predict (vote or average).
*   **Distance Metric:** Euclidean is common; feature scaling is crucial.
*   **Choosing `K`:** Critical hyperparameter; affects model complexity (bias-variance trade-off). Usually tuned via cross-validation.
*   **Pros:** Simple, no training phase, handles multi-class.
*   **Cons:** Computationally expensive at prediction, sensitive to `K`, distance metric, feature scaling, and curse of dimensionality.

For PYQ 6a (May 2023), which likely involved a simple application, understanding the distance calculation and majority voting process is key. 