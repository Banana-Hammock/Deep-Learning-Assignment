import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data preprocessing
df = pd.read_csv("Electric_Production.csv")
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE")

series = df["Value"].values.reshape(-1, 1)

# Min-max scaling
min_val = series.min()
max_val = series.max()
scaled = (series - min_val) / (max_val - min_val)

# Sequence
def build_windows(data, window):
  X, Y = [], []
  for i in range(len(data) - window):
    X.append(data[i:i+window])
    Y.append(data[i+window])
  return np.array(X), np.array(Y)

SEQ_LEN = 24
X, Y = build_windows(scaled, SEQ_LEN)

split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]


train_loader = DataLoader(
  TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(Y_train, dtype=torch.float32)),
  batch_size=32, shuffle=True
)

test_loader = DataLoader(
  TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(Y_test, dtype=torch.float32)),
  batch_size=32, shuffle=False
)

# Models
class SimpleRNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.RNN(1, 64, num_layers=2, batch_first=True)
    self.out = nn.Linear(64, 1)

  def forward(self, x):
    h, _ = self.rnn(x)
    return self.out(h[:, -1])

class SimpleLSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True)
    self.out = nn.Linear(64, 1)

  def forward(self, x):
    h, _ = self.lstm(x)
    return self.out(h[:, -1])

class SimpleGRU(nn.Module):
  def __init__(self):
    super().__init__()
    self.gru = nn.GRU(1, 64, num_layers=2, batch_first=True)
    self.out = nn.Linear(64, 1)

  def forward(self, x):
    h, _ = self.gru(x)
    return self.out(h[:, -1])

# Transformer-like model
class TinyTransformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.embed = nn.Linear(1, 64)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=64, nhead=4, batch_first=True
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    self.out = nn.Linear(64, 1)

  def forward(self, x):
    x = self.embed(x)
    x = self.encoder(x)
    return self.out(x.mean(dim=1))

# Train
def train_network(model, epochs=60):
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_fn = nn.MSELoss()

  losses = []

  for ep in range(epochs):
    model.train()
    total = 0

    for xb, yb in train_loader:
      xb, yb = xb.to(device), yb.to(device)

      optimizer.zero_grad()

      pred = model(xb)
      loss = loss_fn(pred, yb)

      loss.backward()
      optimizer.step()

      total += loss.item()

      avg_loss = total / len(train_loader)
      losses.append(avg_loss)

      if (ep+1) % 20 == 0:
        print(f"Epoch {ep+1}: {avg_loss:.5f}")

  return losses

# Evaluation
def evaluate(model):
  model.eval()
  preds, actual = [], []

  with torch.no_grad():
    for xb, yb in test_loader:
      out = model(xb.to(device)).cpu().numpy()
      preds.extend(out.flatten())
      actual.extend(yb.numpy().flatten())

  # reverse scaling
  preds = np.array(preds) * (max_val - min_val) + min_val
  actual = np.array(actual) * (max_val - min_val) + min_val

  mae = np.mean(np.abs(actual - preds))
  rmse = np.sqrt(np.mean((actual - preds)**2))
  mape = np.mean(np.abs((actual - preds)/actual)) * 100

  return mae, rmse, mape

# future forecast
def forecast(model, steps=120):
  model.eval()
  buffer = scaled[-SEQ_LEN:].flatten().tolist()
  outputs = []

  with torch.no_grad():
    for _ in range(steps):
      x = torch.tensor(buffer[-SEQ_LEN:], dtype=torch.float32).reshape(1, SEQ_LEN, 1).to(device)
      pred = model(x).cpu().item()
      outputs.append(pred)
      buffer.append(pred)

  outputs = np.array(outputs)
  outputs = outputs * (max_val - min_val) + min_val
  return outputs


models = {
  "RNN": SimpleRNN(),
  "LSTM": SimpleLSTM(),
  "GRU": SimpleGRU(),
  "Transformer": TinyTransformer()
}

results = {}

for name, net in models.items():
  print(f"\nTraining {name}")
  train_network(net)

  mae, rmse, mape = evaluate(net)
  future = forecast(net)

  results[name] = (mae, rmse, mape, future)

  print(f"{name} → MAE:{mae:.3f} RMSE:{rmse:.3f} MAPE:{mape:.2f}%")

# Plots and results
future_dates = pd.date_range(
  df["DATE"].max(), periods=121, freq="MS"
)[1:]

plt.figure(figsize=(10,5))

for name, val in results.items():
  plt.plot(future_dates, val[3], label=name)

plt.legend()
plt.title("Future Forecast Comparison")
plt.xlabel("Time")
plt.ylabel("Production")
plt.show()

print("\nModel Performance")
print("-"*40)
for k, v in results.items():
    print(f"{k:10} MAE={v[0]:.3f} RMSE={v[1]:.3f} MAPE={v[2]:.2f}%")