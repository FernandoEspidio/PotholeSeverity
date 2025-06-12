import os
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import onnxruntime as ort
import numpy as np

# ---- Config ----
DATASET_DIR = "archive1"  # unzip Kaggle dataset here
POTH_DIR = os.path.join(DATASET_DIR, "potholes")
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
MODEL_ONNX = "PotholeDetector.onnx"

NUM_EPOCHS = 8
BATCH_SIZE = 16
LR = 1e-4
OPSET = 17
NUM_CLASSES = 2

# ---- Dataset ----
class BinaryPotholeDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.transform = transform
        for label, d in enumerate(root_dirs):
            for img_path in Path(d).glob("*"):
                if img_path.suffix.lower() in [".jpg",".png",".jpeg"]:
                    self.samples.append((str(img_path), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---- Transforms & Loaders ----
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds = BinaryPotholeDataset([POTH_DIR, NORMAL_DIR], transform=tfm)
n_train = int(0.8 * len(ds))
train_ds, val_ds = random_split(ds, [n_train, len(ds)-n_train])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ---- Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ---- Training Loop ----
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

# ---- Export to ONNX ----
model.eval()
dummy = torch.randn(1,3,224,224).to(device)
torch.onnx.export(
    model, dummy, MODEL_ONNX,
    input_names=["input"], output_names=["logits"],
    opset_version=OPSET,
    dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}},
    do_constant_folding=True
)
print(f"Exported ONNX model: {MODEL_ONNX}")

# ---- ONNX Runtime Verification ----
sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
res = sess.run(None, {inp_name: dummy.cpu().numpy()})
print("ONNX inference output shape:", np.array(res[0]).shape)
