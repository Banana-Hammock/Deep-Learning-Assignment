import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
  return x * (1 - x)

# weights
np.random.seed(0)
W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)

lr = 0.1

for epoch in range(5000):
  # Forward pass
  Z1 = np.dot(X, W1)
  A1 = sigmoid(Z1)

  Z2 = np.dot(A1, W2)
  A2 = sigmoid(Z2)

  error = Y - A2

  # Backprop
  dA2 = error * sigmoid_deriv(A2)
  dW2 = np.dot(A1.T, dA2)

  dA1 = np.dot(dA2, W2.T) * sigmoid_deriv(A1)
  dW1 = np.dot(X.T, dA1)

  # Update weights
  W1 += lr * dW1
  W2 += lr * dW2

pred = (A2 > 0.5).astype(int)
print("Predictions:\n", pred)