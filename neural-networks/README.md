# Neural Networks

Implementation of Multilayer Perceptron (MLP) neural networks from scratch and comparison with Convolutional Neural Networks (CNNs) for image classification tasks.

## Dataset

**Kuzushiji-MNIST (KMNIST)**
- 70,000 grayscale images (28x28 pixels) of handwritten Japanese Hiragana characters
- 10 different character classes
- Training set: 60,000 images
- Test set: 10,000 images
- Preprocessing: Images flattened to 784 features and standardized

## Implemented Models

### Multilayer Perceptron (MLP)
- Built from scratch using NumPy
- Configurable architecture with support for:
  - Variable number of hidden layers and units per layer
  - Multiple activation functions (ReLU, Leaky ReLU, Sigmoid)
  - L2 regularization
  - Mini-batch gradient descent optimizer
- Backpropagation implemented with verified gradient calculations

### Convolutional Neural Network (CNN)
- Implemented using Keras/TensorFlow
- Architecture:
  - 3 convolutional layers with 3×3 kernels and ReLU activation
  - Max pooling layers after each convolution
  - 2 fully connected layers with configurable units
  - Dropout for regularization

## Key Findings

### Architecture Effects
- Single hidden layer (256 units) achieved highest accuracy (85.3%)
- Two hidden layers performed slightly worse than one 
- No hidden layers (linear classifier) achieved only ~69% accuracy
- Wider networks generally performed better than deeper ones

### Activation Functions
- ReLU and Leaky ReLU both performed well (85.17% and 84.62%)
- Sigmoid performed poorly (23.91%) due to vanishing gradients in deeper networks

### Regularization Impact
- L2 regularization generally decreased performance as λ increased
- Best performance achieved with no regularization (83.01%)
- Strong regularization (λ ≥ 0.1) caused severe underfitting

### CNN vs MLP Comparison
- CNNs outperformed MLPs in general
- CNNs showed better sample efficiency, achieving ~50% accuracy with only 1000 training examples
- CNN with 256 units in dense layers achieved the best performance

### Sensitivity Analysis
- MLP performance was highly sensitive to batch size (optimal at 16)
- CNN performance was more stable across different batch sizes
- Increasing convolutional filters improved accuracy with diminishing returns beyond 32 filters

## Implementation Details

- Gradient verification confirmed correct backpropagation implementation
- Analytical gradients matched numerical approximations with error < 10⁻¹⁰
- Cross-entropy loss used as cost function for training
- Experiments controlled for batch size, learning rate, and other hyperparameters
