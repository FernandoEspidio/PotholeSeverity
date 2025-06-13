using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PotholeSeverity.Application.Models;

namespace PotholeSeverity.Classifier;

public static class PotholeHighlighter
{
    private const string ModelPath = "PotholeHighlighter.onnx";

    public static List<BoundingBox> Predict(string imagePath)
    {
        DenseTensor<float> inputTensor = ImageUtils.LoadAsTensor(imagePath, size: 640);
        using var session = new InferenceSession(ModelPath);
        string inputName = session.InputMetadata.Keys.First();

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        var boxes = new List<BoundingBox>();
        int boxSize = 6; // x, y, w, h, conf, class
        for (int i = 0; i < output.Length; i += boxSize)
        {
            float conf = output[i + 4];
            if (conf < 0.5) continue;

            boxes.Add(new BoundingBox
            {
                X = output[i],
                Y = output[i + 1],
                Width = output[i + 2],
                Height = output[i + 3],
                Confidence = conf
            });
        }
        return boxes;
    }
}