# Pothole Severity Classification with ML.NET

This project is an AI-powered pothole severity classifier built using **ML.NET** on **Linux (no Visual Studio required)**.
It uses transfer learning to classify road images containing potholes into three severity levels: **low**, **medium**, and **high**.
The model helps road-maintenance teams prioritize repairs based on risk.

## Project Overview

* **Input**: Road-surface images (JPG / PNG)
* **Output**: Predicted pothole severity label – `low`, `medium`, or `high`
* **Framework**: ML.NET
* **Platform**: Linux, using the .NET CLI
* **Model**: Image classification (ResNet-based transfer learning)

## Dataset

* **Source** – Annotated Potholes with Severity Levels by Idan Baruch on Kaggle
  [https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels](https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels)
* **License** – Creative Commons Attribution 4.0 (CC BY 4.0)

**Attribution**
Baruch, I. “Annotated Potholes with Severity Levels.” Kaggle, 2023.
Images were reorganized automatically for training; no other dataset content was modified.

## Installation & Setup (Linux)

### 1. Prerequisites

* .NET SDK 6.0 or later
* Optional: build tools (`gcc`, `make`) if you need to manually install TensorFlow libs

### 2. Create the project & install ML.NET + TensorFlow runtime

```bash
dotnet new console -n PotholeSeverity.Classifier
cd PotholeSeverity.Classifier

# ML.NET packages
dotnet add package Microsoft.ML
dotnet add package Microsoft.ML.ImageAnalytics
dotnet add package Microsoft.ML.Vision

# Native TensorFlow C bindings
dotnet add package SciSharp.TensorFlow.Redist --version 2.3.0
```

### 3. TensorFlow setup (if errors persist)

If you still get `libtensorflow.so not found`:

Option 1 – Use the NuGet package (above), and confirm this file exists after `dotnet build`:

```
bin/Debug/net*/runtimes/linux-x64/native/libtensorflow.so
```

Option 2 – Install TensorFlow manually system-wide:

```bash
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.3.0.tar.gz
sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.3.0.tar.gz
sudo ldconfig
```

Then make sure `/usr/local/lib` is in your `LD_LIBRARY_PATH`.

## Data Preparation

1. Download the Kaggle dataset and extract it to the root of your project:

```
archive/
├── images/        # 717 image files
└── annotations/   # 717 XML files
```

2. Do **not** move or relabel files manually.

Our `Program.cs` parses each XML annotation file and auto-assigns the highest-severity pothole label found in that image.

## Training

Run the project:

```bash
dotnet run
```

Sample output from a real run:

```
Loaded 701 annotated images.
Training… (this can take several minutes on CPU)
✔ Training finished.

Micro-Accuracy : 78.38%
Macro-Accuracy : 54.58%
LogLoss        : 0.7024
Model saved to PotholeSeverityModel.zip

Sample prediction for 'img-1.jpg':
   actual   : medium_pothole
   predicted: medium_pothole
```

## Making Predictions on New Images

Add this to your `Program.cs` or a new console app:

```csharp
var ml = new MLContext();
var model = ml.Model.Load("PotholeSeverityModel.zip", out _);
var engine = ml.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

var output = engine.Predict(new ImageData { ImagePath = "archive/images/test.jpg" });
Console.WriteLine($"Predicted severity: {output.PredictedLabel}");
```

## Acknowledgments

* Dataset provided by Idan Baruch via Kaggle (CC BY 4.0)
* Built with ML.NET, TensorFlow C API, and open-source tooling

## License

* Source code: MIT License
* Dataset: Creative Commons Attribution 4.0 (you must credit the author if reusing)