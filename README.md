# Pothole Severity Classification with ML.NET

This project is an AI-powered pothole severity classifier built using **ML.NET** on **Linux (no Visual Studio required)**. It uses transfer learning to classify road images containing potholes into three severity levels: **low**, **medium**, and **high**. This helps prioritize road maintenance based on pothole danger levels.

## 🔍 Project Overview

* **Input**: Road surface images (JPEG/PNG)
* **Output**: Predicted pothole severity label (`low`, `medium`, or `high`)
* **Framework**: [ML.NET](https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet)
* **Platform**: Linux, using .NET CLI
* **Model Type**: Image classification (transfer learning with pre-trained CNN)

## 📁 Dataset Used

We used the [**Annotated Potholes with Severity Levels**](https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels) dataset by **Idan Baruch**, published on Kaggle.

**License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

**Attribution**:
*Baruch, Idan. "Annotated Potholes with Severity Levels." Kaggle, 2023. [https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels](https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels)*
Images were reorganized by severity class for training. No other content was modified.

## 🛠️ Installation & Setup (Linux)

### Prerequisites

* [.NET SDK 6.0 or later](https://dotnet.microsoft.com/download)
* Optional: [ML.NET CLI](https://learn.microsoft.com/en-us/dotnet/machine-learning/automate-training-with-cli)

### Install ML.NET packages

```bash
dotnet new console -n PotholeSeverityClassifier
cd PotholeSeverityClassifier
dotnet add package Microsoft.ML
dotnet add package Microsoft.ML.ImageAnalytics
dotnet add package Microsoft.ML.Vision
```

## 🗃️ Data Preparation

1. **Download dataset** from Kaggle:
   [https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels](https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels)

2. **Create folders** based on severity levels:

```bash
mkdir -p Data/low Data/medium Data/high
```

3. **Reorganize images** by reading the CSV annotations and placing each image in the correct folder based on its severity label.

4. Final folder structure should look like:

```
PotholeSeverityClassifier/
└── Data/
    ├── low/
    ├── medium/
    └── high/
```

## 🧠 Training the Model

1. Copy the training code into `Program.cs` (from this project).
2. Train the model:

```bash
dotnet run
```

3. The trained model (`PotholeSeverityModel.zip`) will be saved for reuse.

## 📊 Evaluation Output Example

```
MicroAccuracy: 0.89
MacroAccuracy: 0.86
LogLoss: 0.42
```

## 🔍 Making Predictions

Modify the `Program.cs` to predict on new images:

```csharp
var result = predictionEngine.Predict(new ImageData { ImagePath = "test.jpg" });
Console.WriteLine($"Predicted severity: {result.PredictedLabel}");
```

## 🙏 Acknowledgments

* Dataset provided by **Idan Baruch** via Kaggle under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
* ML.NET team and contributors for open-source tooling.

## 📄 License

This project is open-source under the **MIT License**. The dataset used is licensed separately under **CC BY 4.0**. Please ensure you credit dataset authors if redistributing or adapting the data.
