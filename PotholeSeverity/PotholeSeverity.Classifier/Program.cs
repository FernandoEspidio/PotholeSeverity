using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
namespace PotholeSeverity.Classifier;

public static class Program
{
    private static readonly string OnnxModelPath = "PotholeSeverityModel.onnx";
    private static readonly string? imagePath = "sample-image.png"; // Replace with your image path

    private static readonly string[] Classes = { "minor_pothole", "medium_pothole", "major_pothole" };

    public static void Main()
    {
        if (imagePath == null)
        {
            Console.WriteLine("No image found.");
            return;
        }

        Console.WriteLine($"Predicting pothole severity for: {Path.GetFileName(imagePath)}");

        // Load and preprocess image (resize, normalize)
        var inputTensor = LoadAndPreprocess(imagePath); // [1,3,224,224]

        // Run ONNX inference
        using var session = new InferenceSession(OnnxModelPath);
        var inputName = session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        int predictedIndex = Array.IndexOf(output, output.Max());
        string label = Classes[predictedIndex];

        Console.WriteLine($"Predicted severity: {label} (index {predictedIndex})");
    }

    private static DenseTensor<float> LoadAndPreprocess(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        image.Mutate(x => x.Resize(224, 224));

        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 }); // NCHW

        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                var pixel = image[x, y];
                tensor[0, 0, y, x] = ((pixel.R / 255f) - 0.485f) / 0.229f;
                tensor[0, 1, y, x] = ((pixel.G / 255f) - 0.456f) / 0.224f;
                tensor[0, 2, y, x] = ((pixel.B / 255f) - 0.406f) / 0.225f;
            }
        }

        return tensor;
    }
}
