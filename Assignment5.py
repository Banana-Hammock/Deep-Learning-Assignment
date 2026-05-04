import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = "Surface/train/images"
val_path   = "Surface/validation/images"

# Transform pipeline
def get_transforms():
  return transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
  ])

def load_data():
  transform = get_transforms()

  train_set = datasets.ImageFolder(train_path, transform=transform)
  val_set   = datasets.ImageFolder(val_path, transform=transform)

  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)

  return train_loader, val_loader

# final layer for each model
def modify_output_layer(model, arch_name, num_classes=6):
  if arch_name == "resnet":
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)

  elif arch_name == "efficientnet":
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, num_classes)

  elif arch_name == "inception":
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)

  return model

# Freeze layers
def freeze_backbone(model):
  for param in model.parameters():
    param.requires_grad = False
  return model

# Train
def train_model(model, loader, optimizer, loss_fn, arch):
  model.train()
  total_loss = 0
  total_samples = 0

  for inputs, labels in loader:
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()

    outputs = model(inputs)

    if arch == "inception" and isinstance(outputs, tuple):
      main_out, aux_out = outputs
      loss = loss_fn(main_out, labels) + 0.4 * loss_fn(aux_out, labels)
    else:
      loss = loss_fn(outputs, labels)

    loss.backward()
    optimizer.step()

    total_loss += loss.item() * inputs.size(0)
    total_samples += inputs.size(0)

  return total_loss / total_samples

# Evaluation
def evaluate_model(model, loader):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in loader:
      inputs = inputs.to(device)

      outputs = model(inputs)

      if isinstance(outputs, tuple):
        outputs = outputs[0]

      preds = torch.argmax(outputs, dim=1)
      correct += (preds.cpu() == labels).sum().item()
      total += labels.size(0)

  return correct / total

# Full pipeline
def experiment(model, name, train_loader, val_loader):
  print(f"\nRunning: {name}")

  model = freeze_backbone(model)
  model = modify_output_layer(model, name)
  model = model.to(device)

  optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
  )

  criterion = nn.CrossEntropyLoss()

  for epoch in range(10):
    loss = train_model(model, train_loader, optimizer, criterion, name)
    print(f"Epoch {epoch+1} Loss: {loss:.4f}")

  acc = evaluate_model(model, val_loader)
  print(f"Validation Accuracy: {acc:.4f}")

  return acc

# execute
train_loader, val_loader = load_data()

models_dict = {
  "resnet": models.resnet50(weights="DEFAULT"),
  "efficientnet": models.efficientnet_b0(weights="DEFAULT"),
  "inception": models.inception_v3(weights="DEFAULT")
}

results = {}

for key in models_dict:
  results[key] = experiment(models_dict[key], key, train_loader, val_loader)

print("\nSummary:")
for k, v in results.items():
  print(k, "->", v)

best_model = max(results, key=results.get)
print("\nTop Performer:", best_model)