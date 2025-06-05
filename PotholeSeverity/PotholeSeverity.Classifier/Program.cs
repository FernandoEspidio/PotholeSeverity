using System.Xml.Linq;
using Microsoft.ML;
using Microsoft.ML.Vision;

namespace PotholeSeverity.Classifier;

public static class Program
{
    // ---------- EDIT THIS if your folder names differ ----------
    private const string DatasetRoot = "archive";                // folder holding 'images' and 'annotations'
    private const string ImagesSubDir = "images";
    private const string AnnotationsDir = "annotations";
    // -----------------------------------------------------------

    private static readonly Dictionary<string, int> SeverityRank = new()
    {
        {"minor_pothole",  0},
        {"medium_pothole", 1},
        {"major_pothole",  2}
    };

    public static void Main()
    {
        Console.WriteLine("Pothole Severity Classification (ML.NET)\n");

        // 1 Load image paths & labels from XML annotations -----------------------------
        var mlContext = new MLContext(seed: 1);

        string imagesPath = Path.Combine(DatasetRoot, ImagesSubDir);
        string annotPath = Path.Combine(DatasetRoot, AnnotationsDir);

        if (!Directory.Exists(imagesPath) || !Directory.Exists(annotPath))
        {
            Console.Error.WriteLine("✖ Dataset folders not found. Check paths in Program.cs.");
            return;
        }

        var examples = LoadImageData(imagesPath, annotPath);
        Console.WriteLine($"Loaded {examples.Count} annotated images.");

        // 2–3 full preprocessing (label→key + load pixels)
        var data = mlContext.Data.LoadFromEnumerable(examples);
        data = mlContext.Data.ShuffleRows(data, seed: 1);

        var preprocess = mlContext.Transforms.Conversion
                              .MapValueToKey("LabelKey", "Label")
                         .Append(mlContext.Transforms.LoadRawImageBytes(
                              outputColumnName: "Image",
                              imageFolder: imagesPath,
                              inputColumnName: "ImagePath"));

        // 4 train/test split on the *raw* data (so each split still runs preprocess)
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainSet = split.TrainSet;
        var testSet = split.TestSet;

        // 5 ImageClassification trainer options
        var options = new ImageClassificationTrainer.Options
        {
            LabelColumnName = "LabelKey",
            FeatureColumnName = "Image",
            ValidationSet = preprocess.Fit(testSet).Transform(testSet),
            Epoch = 50,
            BatchSize = 10
        };

        // 6 full pipeline = preprocess → trainer → map-back label
        var pipeline = preprocess
                       .Append(mlContext.MulticlassClassification
                               .Trainers.ImageClassification(options))
                       .Append(mlContext.Transforms.Conversion
                               .MapKeyToValue("PredictedLabel", "PredictedLabel"));

        // 7 train
        Console.WriteLine("Training… (this can take several minutes on CPU)");
        var model = pipeline.Fit(trainSet);
        Console.WriteLine("✔ Training finished.\n");

        // 7 Evaluate -------------------------------------------------------------------
        var metrics = mlContext.MulticlassClassification
                        .Evaluate(model.Transform(testSet), "LabelKey");
        Console.WriteLine($"Micro-Accuracy : {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"Macro-Accuracy : {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"LogLoss        : {metrics.LogLoss:F4}\n");

        // 8 Save model -----------------------------------------------------------------
        const string modelFile = "PotholeSeverityModel.zip";
        mlContext.Model.Save(model, trainSet.Schema, modelFile);
        Console.WriteLine($"Model saved to {modelFile}");

        // 9 Demo prediction ------------------------------------------------------------
        var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

        var sample = examples.First();                       // pick first image
        var prediction = predictor.Predict(sample);

        Console.WriteLine($"\nSample prediction for '{Path.GetFileName(sample.ImagePath)}':");
        Console.WriteLine($"   actual   : {sample.Label}");
        Console.WriteLine($"   predicted: {prediction.PredictedLabel}");
    }

    /// <summary>
    /// Parse Pascal-VOC XML files and emit ImageData with the *highest* severity found.
    /// </summary>
    /// <param name="imagesDir">Path to the images folder.</param>
    /// <param name="annotDir">Path to the annotations folder.</param>
    /// <returns>List of ImageData with image paths relative to imagesDir.</returns>
    private static List<ImageData> LoadImageData(string imagesDir, string annotDir)
    {
        var list = new List<ImageData>();

        foreach (var xmlFile in Directory.EnumerateFiles(annotDir, "*.xml"))
        {
            var doc = XDocument.Load(xmlFile);
            var filename = doc.Root?.Element("filename")?.Value;

            if (string.IsNullOrWhiteSpace(filename))
                continue;

            // gather all <name> severity tags in this XML
            var labelsInImg = doc.Descendants("object")
                                 .Select(e => e.Element("name")?.Value)
                                 .Where(n => !string.IsNullOrWhiteSpace(n))
                                 .Select(n => n!.Trim().ToLower())
                                 .ToList();

            if (labelsInImg.Count == 0)
                continue;

            // pick *highest* severity present in that image
            var chosen = labelsInImg
                         .OrderByDescending(l => SeverityRank.GetValueOrDefault(l, -1))
                         .First();

            var fullPath = Path.Combine(imagesDir, filename);
            if (File.Exists(fullPath))
            {
                // image path should be relative to imagesDir
                var relativePath = Path.GetRelativePath(imagesDir, fullPath);
                list.Add(new ImageData { ImagePath = relativePath, Label = chosen });
            }
        }

        return list;
    }
}