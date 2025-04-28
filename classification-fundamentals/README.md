# Classification Fundamentals

Implementation of K-Nearest Neighbors (KNN) and Decision Tree (DT) algorithms from scratch for classification tasks. This project evaluates how these models perform on two different datasets with various parameter configurations.

## Datasets

### Heart Disease Dataset
- **Source**: UCI Machine Learning Repository
- **Task**: Binary classification to predict the presence of heart disease
- **Features**: Medical and demographic attributes
- **Processing**: Transformed target variable into binary (0 = no disease, 1 = disease present)
- **Missing values**: Replaced with mean values for respective features

### Penguins Dataset
- **Task**: Multi-class classification to predict penguin species (Chinstrap, Adelie, Gentoo)
- **Features**: Morphological measurements
- **Processing**: Removed island and sex columns, dropped rows with missing values

## Implemented Models

### K-Nearest Neighbors (KNN)
- Distance-based lazy learning algorithm
- Hyperparameters tested:
  - Number of neighbors (k): 1-20
  - Distance metrics: Euclidean, Manhattan, Cosine
  - Weighting: Standard and weighted variants

### Decision Tree (DT)
- Hierarchical rule-based model
- Hyperparameters tested:
  - Maximum tree depth: 1-20
  - Cost functions: Misclassification, Entropy, Gini Index

## Key Findings

- Decision Trees generally outperformed KNN on both datasets
- For Heart Disease dataset:
  - DT: 0.75 AUROC
  - KNN: 0.71 AUROC
- For Penguins dataset:
  - DT: 97.10% accuracy
  - KNN: 78.26% accuracy
- Feature standardization significantly improved KNN's stability
- Manhattan distance performed best for Heart Disease dataset (0.77 accuracy)
- Cosine distance excelled for Penguins dataset (0.957 accuracy)
- Misclassification cost function performed best for DT on both datasets
- K-fold cross-validation provided more stable accuracy across k-values compared to train-test split

## Feature Importance Analysis

The project includes two approaches for feature importance analysis:
1. Statistical method: Calculating squared differences between class means
2. Decision Tree methods: Feature counts and weighted cost reduction

Top features for Heart Disease dataset using different methods:
- Statistical: thalach, chol, trestbps, age, thal
- Decision Tree: oldpeak, ca, thal, cp, sex

## Usage

The implementation includes the following components:
- KNN and DT class implementations with customizable parameters
- Data preprocessing and feature selection utilities
- Evaluation functions for accuracy and AUROC
- Visualization tools for model comparison

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib

## Performance

Models were evaluated using:
- Train-test split (80-20)
- 5-fold cross-validation
- ROC curve analysis for binary classification
- Accuracy for multi-class classification
