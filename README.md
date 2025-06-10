Absolutely — here is the updated README written directly in Markdown format (without code blocks):

---

# Pothole Severity Classification with ML.NET (ONNX Inference)

This project is an AI-powered pothole severity classifier built using **ML.NET** and **ONNX inference** on **Linux (no Visual Studio required)**.
The model uses a pretrained ResNet and classifies potholes into three severity levels: **low**, **medium**, and **high** — allowing road maintenance teams to prioritize repairs efficiently.

---

## 🗂️ Project Overview

* **Input**: Road-surface images (JPG / PNG)
* **Output**: Predicted pothole severity label – `low`, `medium`, or `high`
* **Inference**: ONNX model (trained with PyTorch)
* **App Runtime**: .NET 6+ CLI, cross-platform

---

## 📦 Dataset

* **Source** – Annotated Potholes with Severity Levels by Idan Baruch on Kaggle
  [https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels](https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels)
* **License** – Creative Commons Attribution 4.0 (CC BY 4.0)

**Attribution**
Baruch, I. “Annotated Potholes with Severity Levels.” Kaggle, 2023.
Images were reorganized automatically for training; no other dataset content was modified.

---

## 🚀 Installation & Setup (Linux)

### 1. Prerequisites

* Python 3.10+ with `torch`, `torchvision`, `onnx`, `onnxruntime`
* .NET 6.0+ SDK installed (`dotnet --version` to check)

### 2. Clone and prepare the dataset

Download the dataset and place it like this:

```
PotholeSeverity/
├── PotholeSeverity.Classifier/
├── PotholeSeverity.Application/
└── archive/
    ├── images/        # 717 JPGs
    └── annotations/   # 717 XMLs
```

---

## 🧠 Model Training (Python → ONNX)

Before you run any .NET code, you **must generate the ONNX model file** using Python:

### 1. Set up Python environment

From the `PotholeSeverity/PotholeSeverityClassifier` folder:

```bash
pip install torch torchvision onnx onnxruntime pillow lxml
```

### 2. Run the training script

```bash
python3 train_pothole_classifier.py
```

This will generate the ONNX model at:

```
PotholeSeverity/PotholeSeverity.Classifier/PotholeSeverityModel.onnx
```

✅ You are now ready to run the C# ONNX-powered inference.

To use the service project API, you will have to copy the ONNX model to the service project folder:

```bash
cp PotholeSeverity.Classifier/PotholeSeverityModel.onnx PotholeSeverity.Application/Models/
cp PotholeSeverity.Classifier/PotholeSeverityModel.onnx PotholeSeverity.Api/Models/
```

---

## 🖥️ Running the Inference App (C#)

From the project root:

```bash
cd PotholeSeverity/PotholeSeverity.Classifier
dotnet run
```

The app will:

* Load the ONNX model
* Preprocess a sample image
* Output the predicted severity level

Example output:

```
Predicting pothole severity for: pothole_001.jpg
Predicted severity: major_pothole (index 2)
```

---

## 🧪 Prediction on Custom Images

To classify another image (e.g., `test.jpg`), drop it into `archive/images/`, then either edit `Program.cs` to point to it or extend the app to read from arguments.

---

## 🛠️ Notes

* The ONNX model uses standard PyTorch preprocessing: Resize → Normalize → NCHW.
* Image preprocessing is done using [ImageSharp](https://github.com/SixLabors/ImageSharp) for full Linux/macOS support.
* The app **does not** train any models — it performs pure inference using the ONNX model.

---

## ✅ Acknowledgments

* Dataset provided by Idan Baruch via Kaggle (CC BY 4.0)
* Built using PyTorch, ONNX Runtime, .NET CLI, and open-source tooling

---

## 📝 License

* Source code: MIT License
* Dataset: Creative Commons Attribution 4.0 (CC BY 4.0)

---

Let me know if you want this saved as a file or updated directly in your repo.
