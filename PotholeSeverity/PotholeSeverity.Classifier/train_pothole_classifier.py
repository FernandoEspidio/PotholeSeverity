import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import onnx
import onnxruntime as ort
import numpy as np

# ---- Config ----
DATASET_DIR = "archive"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
ANNOT_DIR = os.path.join(DATASET_DIR, "annotations")
MODEL_OUT = "PotholeSeverityModel.onnx"

# ---- Pothole Dataset (reads Pascal VOC format) ----
SEVERITY_RANK = {"minor_pothole": 0, "medium_pothole": 1, "major_pothole": 2}

class PotholeDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.classes = sorted(SEVERITY_RANK.keys())
        self.transform = transform
        for xml_file in Path(ANNOT_DIR).glob("*.xml"):
            tree = ET.parse(xml_file)
            fname = tree.findtext("./filename")
            labels = [obj.findtext("name").strip().lower()
                      for obj in tree.findall("./object")]
            if not fname or not labels:
                continue
            label = max(labels, key=lambda l: SEVERITY_RANK.get(l, -1))
            path = Path(IMAGE_DIR) / fname
            if path.exists():
                self.samples.append((str(path), self.classes.index(label)))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ---- Preprocessing & Loaders ----
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

dataset = PotholeDataset(transform=tfm)
train_size = int(0.8 * len(dataset))
train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
val_loader = DataLoader(val_set, batch_size=10)

# ---- Model ----
model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.train()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# ---- Training ----
print("Training started...")
for epoch in range(10):  # reduce for quicker test runs
    for X, y in train_loader:
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}: Val Accuracy = {acc:.2%}")
    model.train()

# ---- Export to ONNX ----
print(f"Exporting model to {MODEL_OUT}")
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.eval(),
    dummy_input,
    MODEL_OUT,
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
)

# ---- Verify with ONNXRuntime ----
print("Verifying ONNX model...")
session = ort.InferenceSession(MODEL_OUT, providers=["CPUExecutionProvider"])
onnx_output = session.run(None, {"input": dummy_input.numpy()})
print("ONNX model works. Output shape:", np.array(onnx_output).shape)
