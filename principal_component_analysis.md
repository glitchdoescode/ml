# Principal Component Analysis (PCA): Reducing Dimensionality (PYQ 3b - 2022, PYQ 6b - CBGS)

## 1. What is Principal Component Analysis (PCA)?

**Principal Component Analysis (PCA)** is one of the most popular **unsupervised dimensionality reduction** techniques in machine learning and statistics. It's used to transform a dataset with a large number of variables (features) into a smaller set of new variables, called **principal components (PCs)**, while retaining as much of the original information (variance) as possible.

**Core Idea:** PCA aims to find the directions (principal components) in the data that capture the maximum amount of variance. These components are orthogonal (uncorrelated) to each other. By projecting the original data onto a lower-dimensional subspace formed by the top principal components (those that explain the most variance), we can reduce the dimensionality of the data.

**Analogy: Summarizing a Story**
Imagine you have a very long and detailed story (high-dimensional data). PCA is like trying to find the main plotlines (principal components) that capture the essence of the story. You might lose some minor details (less important variance), but you get a much shorter and more manageable summary (lower-dimensional data) that still conveys most of the important information.

## 2. Why Use PCA? (Purpose & Goals)

PCA is used for several reasons:

*   **Dimensionality Reduction:** This is the primary goal.
    *   **Simplify Models:** Fewer input features can lead to simpler and faster models.
    *   **Reduce Computational Cost:** Training models and making predictions is faster with fewer dimensions.
    *   **Overcome the Curse of Dimensionality:** In very high-dimensional spaces, data becomes sparse, and statistical models can perform poorly. PCA can mitigate this.
*   **Data Visualization:** By reducing data to 2 or 3 principal components, high-dimensional data can be plotted and visually explored to identify patterns or clusters.
*   **Noise Reduction:** Principal components associated with very low variance might represent noise in the data. Discarding these can lead to a cleaner dataset.
*   **Feature Extraction / Feature Engineering:** The principal components are new, uncorrelated features derived from linear combinations of the original features. These can sometimes be more informative for subsequent learning tasks.
*   **Collinearity Handling:** PCA transforms correlated original features into uncorrelated principal components.

## 3. Key Concepts in PCA

### a) Variance and Covariance
*   **Variance:** Measures the spread or dispersion of a single variable around its mean. Higher variance means more spread out data.
*   **Covariance:** Measures how two variables change together (co-vary). A positive covariance indicates that as one variable increases, the other tends to increase. A negative covariance indicates that as one increases, the other tends to decrease. A covariance near zero suggests little linear relationship.
*   PCA utilizes the **covariance matrix** of the data to find the directions of maximum variance.

### b) Eigenvectors and Eigenvalues
These are concepts from linear algebra that are crucial to PCA.
*   For a given square matrix (like the covariance matrix of the data), an **eigenvector** is a non-zero vector that, when multiplied by the matrix, results in a scaled version of itself. The scaling factor is the **eigenvalue**.
    `Covariance_Matrix * Eigenvector = Eigenvalue * Eigenvector`
*   **Role in PCA:**
    *   **Eigenvectors of the Covariance Matrix:** These vectors define the **directions** of the principal components. They point in the directions of maximum variance in the data.
    *   **Eigenvalues:** The eigenvalue associated with each eigenvector indicates the **amount of variance** in the data that is captured by that eigenvector (i.e., by the corresponding principal component).

### c) Principal Components (PCs)
*   **Definition:** The principal components are new, uncorrelated variables that are linear combinations of the original variables.
*   **Construction:** They are constructed such that:
    *   The **first principal component (PC1)** is the direction in the data that explains the largest possible variance.
    *   The **second principal component (PC2)** is orthogonal (perpendicular) to PC1 and explains the largest possible remaining variance.
    *   This continues for subsequent components, each being orthogonal to all previous ones and capturing the maximum remaining variance.
*   The principal components are the eigenvectors of the data's covariance matrix, ordered by their corresponding eigenvalues (from largest to smallest).

## 4. Steps Involved in PCA

1.  **Standardize the Data (Feature Scaling):**
    *   PCA is sensitive to the scale of the original features. If features have vastly different scales (e.g., one feature in meters and another in kilometers), the feature with the larger scale can dominate the variance calculation and thus the principal components.
    *   Therefore, it's standard practice to scale each feature to have zero mean and unit variance (z-score normalization) before applying PCA.
        `z = (x - μ) / σ` (where `μ` is the mean and `σ` is the standard deviation of the feature)

2.  **Compute the Covariance Matrix:**
    *   Calculate the covariance matrix of the standardized data. If you have `d` features, the covariance matrix will be a `d x d` symmetric matrix.
    *   The diagonal elements of this matrix represent the variances of each feature, and the off-diagonal elements represent the covariances between pairs of features.

3.  **Compute Eigenvectors and Eigenvalues (Eigendecomposition):**
    *   Perform eigendecomposition on the covariance matrix to obtain its eigenvectors and corresponding eigenvalues.
    *   Each eigenvector will have the same dimensionality as the original data (`d`), and there will be `d` eigenvectors and `d` eigenvalues.

4.  **Sort Eigenvectors by Eigenvalues:**
    *   Sort the eigenvectors in descending order based on the magnitude of their corresponding eigenvalues. The eigenvector with the largest eigenvalue is the first principal component (PC1), the one with the second largest eigenvalue is PC2, and so on.

5.  **Select Principal Components (Choose `k`):**
    *   Decide how many principal components (`k`) to keep for the new, lower-dimensional feature subspace. `k` will be less than the original number of dimensions `d`.
    *   This choice can be based on:
        *   **A predefined number of dimensions:** If you know you want to reduce to, say, 2 or 3 dimensions for visualization.
        *   **Desired amount of cumulative explained variance:** Calculate the proportion of variance explained by each PC (`Eigenvalue_i / Sum_of_all_Eigenvalues`). Then, sum these proportions cumulatively and choose `k` such that, for example, 90%, 95%, or 99% of the total variance is retained.
        *   **Scree Plot:** Plot the eigenvalues in descending order. Look for an "elbow" point where the eigenvalues start to level off. The components before the elbow are often considered the most significant.

6.  **Form the Projection Matrix (W):**
    *   Create a matrix `W` whose columns are the selected top `k` eigenvectors (the ones corresponding to the `k` largest eigenvalues).
    *   This matrix `W` will have dimensions `d x k`.

7.  **Transform the Data (Project onto New Subspace):**
    *   Project the original standardized data (`X_standardized`, which is `n x d`, where `n` is number of samples) onto the new `k`-dimensional subspace by multiplying it with the projection matrix `W`:
        `X_transformed = X_standardized * W`
    *   The resulting `X_transformed` will be an `n x k` matrix, representing the data in terms of the `k` principal components.

## 5. Interpreting Principal Components

*   Each principal component represents a direction in the original feature space along which the data varies the most (after accounting for previous components).
*   The loadings (coefficients of the linear combination that forms a PC from original features) can sometimes give an indication of which original features contribute most to a particular PC. However, interpretation can become difficult if PCs are combinations of many original features.

## 6. Advantages of PCA

*   **Reduces Dimensionality:** Simplifies data and models.
*   **Identifies Directions of Maximum Variance:** Captures the most important patterns in terms of data spread.
*   **Uncorrelated Components:** The resulting principal components are orthogonal and thus uncorrelated, which can be beneficial for some machine learning algorithms that are sensitive to multicollinearity.
*   **Noise Reduction:** Can filter out noise by discarding components with low variance.
*   **Improved Model Performance:** By reducing dimensionality and noise, PCA can sometimes lead to better generalization and performance of learning algorithms, and reduce overfitting.

## 7. Disadvantages of PCA

*   **Assumes Linearity:** PCA is based on linear transformations and linear correlations. It may not perform well if the underlying structure of the data is highly non-linear (Kernel PCA can address this).
*   **Loss of Interpretability:** Principal components are linear combinations of the original features and may not have clear, intuitive meanings compared to the original features.
*   **Sensitive to Data Scaling:** As mentioned, features must be standardized before PCA.
*   **Information Loss:** Dimensionality reduction inevitably leads to some information loss, though PCA tries to minimize this by retaining maximum variance.
*   **Variance Might Not Mean Importance:** PCA prioritizes variance. In some cases (e.g., certain classification tasks), directions with lower variance might still be important for distinguishing classes. PCA is unsupervised and doesn't consider class labels.
*   **Computationally Intensive for Very High Dimensions:** Calculating the covariance matrix and its eigendecomposition can be demanding for extremely high-dimensional data (though techniques like Randomized PCA exist).

## 8. Applications of PCA

*   **Image Compression:** Reducing the number of dimensions needed to represent images.
*   **Face Recognition (Eigenfaces):** PCA is used to extract key features from face images.
*   **Bioinformatics:** Analyzing gene expression data, identifying patterns in biological datasets.
*   **Finance:** Analyzing stock data, risk management.
*   **Data Visualization:** Reducing data to 2D or 3D for plotting.
*   **Preprocessing for other ML algorithms.**

## 9. Summary for Exams (PYQ 3b - 2022, PYQ 6b - CBGS)

*   **PCA Goal:** An **unsupervised dimensionality reduction** technique.
*   **Core Idea:** Transform data to a lower-dimensional space by finding **principal components (PCs)** – new, uncorrelated variables that capture the **maximum variance** in the data.
*   **Key Steps:**
    1.  Standardize data.
    2.  Compute covariance matrix.
    3.  Calculate eigenvectors and eigenvalues of the covariance matrix.
    4.  Select top `k` eigenvectors (based on eigenvalues/explained variance) to form PCs.
    5.  Project data onto these `k` PCs.
*   **Eigenvectors & Eigenvalues:** Eigenvectors define the directions of PCs; eigenvalues indicate the variance explained by each PC.
*   **Purpose:** Reduce dimensions, visualize data, remove noise, create uncorrelated features.
*   **Limitations:** Assumes linearity, PCs can be hard to interpret, sensitive to scaling.

Understanding that PCA finds new axes (principal components) that are orthogonal and capture decreasing amounts of variance is key. 