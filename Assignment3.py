import numpy as np
import pandas as pd

df = pd.read_csv("glass.csv")

# binary classification
df["Label"] = df["Type"].apply(lambda x: 0 if x <= 4 else 1)

X = df.drop(["Type", "Label"], axis=1).values
Y = df["Label"].values.reshape(-1,1)

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

W = np.zeros((X.shape[1], 1))
b = 0
lr = 0.01
epochs = 5000

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

# train
for _ in range(epochs):
  Z = np.dot(X, W) + b
  A = sigmoid(Z)

  error = A - Y

  dW = np.dot(X.T, error) / len(X)
  db = np.mean(error)

  W -= lr * dW
  b -= lr * db

pred = sigmoid(np.dot(X, W) + b)
pred_class = (pred > 0.5).astype(int)

accuracy = np.mean(pred_class == Y)
print("Accuracy:", accuracy)