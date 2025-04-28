# Linear Models for Classification

Implementation of linear regression, logistic regression, and multinomial logistic regression from scratch for binary and multi-class classification tasks.

## Datasets

### Breast Cancer Wisconsin Dataset
- **Source**: UCI Machine Learning Repository
- **Task**: Binary classification to predict tumor malignancy (0 = benign, 1 = malignant)
- **Features**: 30 numerical attributes extracted from cell nuclei images (via fine needle aspiration)
- **Processing**: Removed 5 low-importance features based on regression coefficient analysis

### Wine Recognition Dataset
- **Source**: UCI Machine Learning Repository
- **Task**: Multi-class classification to identify wine cultivars (3 classes)
- **Features**: 13 numerical features representing chemical composition
- **Processing**: One-hot encoded class labels and standardized numerical features

## Implemented Models

### Multiple Linear Regression (MLR)
- Uses least squares to estimate regression coefficients
- Transforms outputs to probabilities using sigmoid function for binary classification
- Rounds and clips predictions for multi-class tasks

### Logistic Regression
- Binary classification using sigmoid activation
- Cross-entropy loss optimization via gradient descent
- Numerical stability through gradient clipping

### Multinomial Logistic Regression
- Multi-class classification using softmax function
- Early stopping based on loss convergence
- Optimized with appropriate learning rates

## Key Findings

- Logistic regression outperformed linear regression in binary classification with AUROC of 1.00 vs. 0.99
- In multi-class classification, both multivariate linear regression and multi-class logistic regression achieved perfect accuracy (100%)
- Gradient checking confirmed the correctness of our gradient computations with differences of 3.38×10⁻⁸ for binary and 1.04×10⁻¹⁴ for multi-class
- Both models generally agreed on feature importance direction but differed in coefficient magnitudes
- Logistic regression coefficients showed stronger feature discrimination with larger magnitudes
- Both models outperformed KNN and Decision Tree algorithms on the binary classification task

## Feature Importance Analysis

- Identified most influential features through regression coefficients
- For Breast Cancer: concave_points1, perimeter1, area1, radius1, area2
- For Wine Dataset: 
  - Class 1: Proline, Flavanoids, Alcohol
  - Class 2: Alcalinity of ash, Hue
  - Class 3: Color intensity, Malic acid

## Model Evaluation

Models were evaluated using:
- Area Under the ROC Curve (AUROC) for binary classification
- Accuracy for multi-class classification
- Train-validation-test split (70-15-15) for hyperparameter optimization

## Implementation Details

- All models include options for bias terms
- Implemented proper validation techniques to prevent overfitting
- Binary logistic regression achieved optimal performance with learning rate 0.001 and 10,000 iterations
- Multi-class logistic regression performed best with learning rate 1.0 and 1,000 iterations
