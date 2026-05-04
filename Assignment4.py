import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# one hot encoding
def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        result[i, y[i]] = 1
    return result

y_train = one_hot(y_train)
y_test = one_hot(y_test)

# weights
np.random.seed(1)
W1 = np.random.randn(784, 128) * 0.01
W2 = np.random.randn(128, 64) * 0.01
W3 = np.random.randn(64, 10) * 0.01

lr = 0.1

def relu(x):
  return np.maximum(0, x)

def relu_deriv(x):
  return (x > 0).astype(int)

def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
  return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Train
for epoch in range(10):

  # Forward
  Z1 = np.dot(X_train, W1)
  A1 = relu(Z1)

  Z2 = np.dot(A1, W2)
  A2 = relu(Z2)

  Z3 = np.dot(A2, W3)
  A3 = softmax(Z3)

  # Backprop
  dZ3 = A3 - y_train
  dW3 = np.dot(A2.T, dZ3)

  dZ2 = np.dot(dZ3, W3.T) * relu_deriv(A2)
  dW2 = np.dot(A1.T, dZ2)

  dZ1 = np.dot(dZ2, W2.T) * relu_deriv(A1)
  dW1 = np.dot(X_train.T, dZ1)

  # Update weights
  W1 -= lr * dW1
  W2 -= lr * dW2
  W3 -= lr * dW3

# Test
Z1 = np.dot(X_test, W1)
A1 = relu(Z1)
Z2 = np.dot(A1, W2)
A2 = relu(Z2)
Z3 = np.dot(A2, W3)
A3 = softmax(Z3)

pred = np.argmax(A3, axis=1)
true = np.argmax(y_test, axis=1)

accuracy = np.mean(pred == true)
print("Accuracy:", accuracy)