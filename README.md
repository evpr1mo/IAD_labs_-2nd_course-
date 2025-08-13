# Intelligent Data Analysis (IAD) Course Labs

This repository contains the laboratory works for the **Intelligent Data Analysis (IAD)** course. The labs focus on practical skills in data analysis using Python, including market basket analysis, classification, regression, neural networks, clustering, and ensemble methods. All implementations are based on libraries such as NumPy, Scikit-Learn, and Matplotlib.

## Prerequisites

- Python 3.x
- Required libraries: `numpy`, `scikit-learn`, `matplotlib`, `pandas` (install via `pip install -r requirements.txt` if a requirements file is provided)

## Table of Contents

- [Lab 1: Acquiring Skills in Working with the Python Environment](#lab-1-acquiring-skills-in-working-with-the-python-environment)
- [Lab 2: Building and Evaluating Quality of Classification and Regression Models Using Scikit-Learn](#lab-2-building-and-evaluating-quality-of-classification-and-regression-models-using-scikit-learn)
- [Lab 3: Classification and Regression Based on Multilayer Perceptron in Scikit-Learn](#lab-3-classification-and-regression-based-on-multilayer-perceptron-in-scikit-learn)
- [Lab 4: Building and Evaluating Quality of Clustering Models in Scikit-Learn](#lab-4-building-and-evaluating-quality-of-clustering-models-in-scikit-learn)
- [Lab 5: Building and Evaluating Ensembles of Classification and Regression Models Using Scikit-Learn](#lab-5-building-and-evaluating-ensembles-of-classification-and-regression-models-using-scikit-learn)

## Lab 1: Acquiring Skills in Working with the Python Environment

### Objective
To gain practical skills in working with the Python environment through market basket analysis.

### Task Description

<img width="881" height="659" alt="image" src="https://github.com/user-attachments/assets/aa0b27e5-2528-44a5-8c28-204544594703" />

### Implementation Notes
Implement the algorithm in Python using basic libraries like NumPy for efficiency. Avoid external libraries for the core logic.

## Lab 2: Building and Evaluating Quality of Classification and Regression Models Using Scikit-Learn

### Objective
To build and evaluate the quality of models including decision trees, support vector machines, logistic regression, and naive Bayes for classification and regression using the Scikit-Learn library.

### Task Description (Variant 25)
Build naive Bayes classification models under the following assumptions:
- Data in each class follows a normal distribution without covariance between dimensions; use `sklearn.naive_bayes.GaussianNB`.
- Data in each class follows a multinomial distribution; use `sklearn.naive_bayes.MultinomialNB`.
- For each model, calculate posterior probabilities for a test example using the `predict_proba` method.

### Initial Data
- **Dataset (a)**:
  ```python
  from sklearn.datasets import make_blobs
  import numpy as np
  X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
  rng = np.random.RandomState(13)
  X_stretched = np.dot(X, rng.randn(2, 2))
  ```
- **Dataset (b)**:
  ```python
  import numpy as np
  np.random.seed(0)
  X = np.random.randn(300, 2)
  Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
  ```

> **Note**: Dataset (b) was adjusted for MultinomialNB to a more suitable polynomial set with feature indicators for better demonstration.

### Procedure
1. Visualize the initial data graphically.
2. Split the data into training and validation sets.
3. Build the specified classification models on the training set.
4. Visualize the models (e.g., display part of a decision tree).
5. Perform predictions based on the built models.
6. Evaluate if overfitting occurs for each model.
7. Calculate additional model results, such as posterior probabilities (as per variant).
8. In classification tasks, graphically plot decision boundaries for each model.
9. In classification tasks, calculate the following quality metrics for each model on both training and validation sets:
   - Confusion matrix
   - Precision
   - Recall
   - F1 score
   - Precision-Recall (PR) curve
   - ROC curve
   - AUC score
10. Perform grid search for hyperparameter tuning.
11. Draw conclusions on model performance and select the best model based on quality metrics.
12. Train models on subsets of the training data and assess the impact of training set size on model quality.
13. Research both datasets following the above steps. Optionally, use a custom dataset (with instructor approval).

## Lab 3: Classification and Regression Based on Multilayer Perceptron in Scikit-Learn

### Objective
To build and evaluate the quality of classification models using the Scikit-Learn library (`MLPClassifier`) and multilayer perceptrons.

### Task Description
Use `sklearn.neural_network.MLPClassifier` for classification tasks. Start with a single-layer model and determine if it is sufficient for the data. Implement dynamic addition of neurons to the hidden layer and check how many neurons are needed in a single-layer model for satisfactory task resolution.

### Initial Data
Same as Lab 2:
- **Dataset (a)**: Stretched blobs from `make_blobs`.
- **Dataset (b)**: Random features with binary labels.

### Procedure
Follow the same steps as in Lab 2.

## Lab 4: Building and Evaluating Quality of Clustering Models in Scikit-Learn

### Objective
To build and evaluate the quality of clustering models using Scikit-Learn.

### Task Description (Variant 25)
Use the agglomerative algorithm `AgglomerativeClustering`. Investigate distance calculation methods between clusters: `ward`, `single`, `average`, `complete`.  
**Quality Metrics**: Estimated number of clusters, Adjusted Rand Index (ARI), V-measure.  
Build distance matrices between clusters using `metrics.pairwise_distances`.  
Assess if the partitioning is stable after removing individual objects.

### Initial Data
- **Dataset (a)**: `sklearn.datasets.make_moons`
- **Dataset (b)**: `sklearn.datasets.load_iris`

### Procedure
1. Visualize the initial data graphically.
2. Build the clustering model as per the variant.
3. Perform clustering on the data based on the model.
4. Visualize the cluster partitioning (e.g., using different colors).
5. Calculate clustering time and evaluate method speed on large datasets (e.g., increasing data points to 100,000+).
6. Build alternative models using different distance functions where applicable.
7. For each alternative model, calculate clustering quality metrics from `sklearn.metrics`:
   - Estimated number of clusters
   - Adjusted Rand Index
   - V-measure
8. Perform informal analysis of clustering results (as per variant): Assess stability after removing individual objects.
9. Perform the above steps for both datasets of different shapes.
10. Draw conclusions on model performance, speed, and select the best clustering model for each dataset based on metrics and informal methods.

## Lab 5: Building and Evaluating Ensembles of Classification and Regression Models Using Scikit-Learn

### Objective
To build and evaluate ensembles of models for classification and regression using Scikit-Learn.

### Task Description (Variant 25)
Use `BaggingClassifier`. Consider different values of parameters such as `max_samples`, `bootstrap`, and `n_estimators`.

> **Note**: The description mentions `learning_rate` and `algorithm`, which may apply to boosting methods like AdaBoost; adjust accordingly for bagging.

### Initial Data
- **Dataset (a)**:
  ```python
  from sklearn.datasets import make_circles
  X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
  ```

### Procedure
1. Visualize the initial 2D data graphically.
2. Split the data into training, validation, and test sets. Use validation for hyperparameter tuning and test for final quality evaluation.
3. Build ensembles using the specified methods:
   - `BaggingClassifier` with variations in `max_samples`, `bootstrap`, `n_estimators`.
   - As base estimators, use one or more default models: decision trees, logistic regression, SVM, etc.
   - Plot graphs of ensemble and individual model quality metrics vs. `n_estimators` (e.g., `accuracy_score`, `f1_score`, or `zero_one_loss`).
   - Evaluate ensemble quality using out-of-bag (OOB) samples for bagging-based ensembles.
4. In classification tasks, provide examples of decision boundaries for individual models and the ensemble.
5. Calculate bias and variance for individual models and the ensemble.
6. Compare training time of the ensemble vs. individual models.
7. Draw conclusions: Assess if the ensemble performs better than individual models on the given data.
