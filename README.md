# Pothole Severity Classification with ML.NET (ONNX Inference)

This project is an AI-powered pothole severity and detection classifier built using **ML.NET** and **ONNX inference** on **Linux (no Visual Studio required)**.
It contains two models:

1. A **severity classifier** for prioritizing road maintenance.
2. A **binary detector** to check if a pothole is present or not.

Both models are trained in Python with PyTorch and exported to ONNX for .NET-based inference.

---

## 🗂️ Project Overview

* **Input**: Road-surface images (JPG / PNG)
* **Output**:

  * Severity mode: `low`, `medium`, `high`
  * Binary mode: `pothole` or `normal`
* **Inference**: ONNX models (trained with PyTorch)
* **App Runtime**: .NET 6+ CLI, cross-platform

---

## 📦 Datasets

### 🔸 Severity Classification

* **Source** – Annotated Potholes with Severity Levels by Idan Baruch on Kaggle
  [https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels](https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels)

### 🔸 Binary Pothole Detection

* **Source** – Pothole Detection Dataset by Atulya Kumar on Kaggle
  [https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)

---

## 🚀 Installation & Setup (Linux)

### 1. Prerequisites

* Python 3.10+ with `torch`, `torchvision`, `onnx`, `onnxruntime`
* .NET 6.0+ SDK installed (`dotnet --version` to check)

### 2. Directory structure

Place the datasets like this:

```
PotholeSeverity/
├── PotholeSeverity.Classifier/
├── PotholeSeverity.Application/
├── archive/         # severity dataset
│   ├── images/
│   └── annotations/
└── archive1/        # binary detection dataset
    ├── potholes/
    └── normal/
```

---

## 🧠 Model Training (Python → ONNX)

### 🔹 Severity Classifier

From the `PotholeSeverityClassifier` folder:

```bash
pip install torch torchvision onnx onnxruntime pillow lxml
python3 train_pothole_classifier.py
```

This produces:

```
PotholeSeverity.Classifier/PotholeSeverityModel.onnx
```

### 🔹 Binary Detector (Pothole Presence)

For basic detection (pothole vs. no pothole):

```bash
python3 train_pothole_detector.py
```

This produces:

```
PotholeSeverity.Classifier/PotholeDetector.onnx
```

✅ Once models are trained, you're ready for inference.

To use these with .NET:

```bash
cp PotholeSeverity.Classifier/PotholeSeverityModel.onnx PotholeSeverity.Application/Models/
cp PotholeSeverity.Classifier/PotholeDetector.onnx PotholeSeverity.Application/Models/
```

(Repeat for `PotholeSeverity.Api/Models/` if needed)

---

## 🖥️ Running the Inference App (C#)

From the project root:

```bash
cd PotholeSeverity/PotholeSeverity.Application
dotnet run
```

The app will:

* Load the specified ONNX model
* Preprocess an input image
* Output the predicted class

Example output (binary mode):

```
Predicting pothole presence for: sample.jpg
Prediction: pothole (index 1)
```

---

## 🧪 Prediction on Custom Images

To classify custom images, place them into the appropriate dataset folder (`archive/images/` or `archive1/potholes/`) and update the path in the application or extend it to accept arguments.

---

## 🛠️ Notes

* The models use standard ResNet18 with transfer learning.
* Preprocessing: Resize to 224×224, normalize to ImageNet stats.
* Inference is handled via ONNX Runtime in both Python and C#.
* Image loading in .NET uses [ImageSharp](https://github.com/SixLabors/ImageSharp) for cross-platform compatibility.

---

## ✅ Acknowledgments

* Dataset (severity): Idan Baruch, Kaggle (CC BY 4.0)
* Dataset (binary): Atulya Kumar, Kaggle (CC BY 4.0)
* Tools: PyTorch, ONNX, ML.NET, ONNX Runtime, .NET CLI

---

## 📝 License

* Source code: MIT License
* Datasets: Creative Commons Attribution 4.0 (CC BY 4.0)
