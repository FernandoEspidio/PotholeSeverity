import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import onnx

# ---- Config ----
IMAGE_DIR = Path("archive/images")
ANNOT_DIR = Path("archive/annotations")
MODEL_OUT = "PotholeHighlighter.onnx"
NUM_CLASSES = 4  # 3 classes + background
NUM_EPOCHS = 3
BATCH_SIZE = 2

CLASS_NAMES = ["minor_pothole", "medium_pothole", "major_pothole"]
CLASS_TO_IDX = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}  # +1 because 0 = background

# ---- Dataset ----
class PotholeDetectionDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transforms=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transforms = transforms
        self.image_files = list(img_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        annot_path = self.annot_dir / (img_path.stem + ".xml")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes = []
        labels = []
        tree = ET.parse(annot_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in CLASS_TO_IDX:
                continue
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)

        return img, target

# ---- Load Data ----
dataset = PotholeDetectionDataset(IMAGE_DIR, ANNOT_DIR)
train_len = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ---- Model ----
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- Training ----
print("Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {losses.item():.4f}")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Total Loss: {epoch_loss:.4f}")

# ---- Export to ONNX ----
print(f"\nExporting model to {MODEL_OUT}...")
model.eval()
dummy_input = [torch.randn(3, 720, 720).to(device)]
torch.onnx.export(
    model, dummy_input, MODEL_OUT,
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],
    dynamic_axes={"input": {0: "batch"}},
    opset_version=11
)
print("✅ ONNX export complete.")
