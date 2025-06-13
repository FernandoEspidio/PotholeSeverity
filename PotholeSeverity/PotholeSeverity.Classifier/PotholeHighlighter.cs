using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PotholeSeverity.Application.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics;

namespace PotholeSeverity.Classifier;

public static class PotholeHighlighter
{
    private const string ModelPath = "PotholeHighlighter.onnx";
    private const float ConfidenceThreshold = 0.5f;

    public static List<BoundingBox> Predict(string imagePath)
    {
        // Load and prepare the image - ensure we're getting a 3D tensor (C,H,W) not 4D (batch,C,H,W)
        using var session = new InferenceSession(ModelPath);
        string inputName = session.InputMetadata.Keys.First();

        // Get expected input shape from the model
        var inputMeta = session.InputMetadata[inputName];

        // Load image as tensor with correct shape (CHW, not NCHW)
        DenseTensor<float> inputTensor = ImageUtils.LoadImageAs3DTensor(imagePath);

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);

        // Get detections as a 2D tensor [num_detections, 6]
        // Each row: [x1, y1, x2, y2, confidence, class]
        var detections = results.First().AsTensor<float>();

        var boxes = new List<BoundingBox>();

        // Get the original image dimensions to scale the normalized coordinates
        var originalImage = Image.Load(imagePath);
        float imgWidth = originalImage.Width;
        float imgHeight = originalImage.Height;

        for (int i = 0; i < detections.Dimensions[0]; i++)
        {
            float confidence = detections[i, 4];
            if (confidence < ConfidenceThreshold) continue;

            float x1 = detections[i, 0];
            float y1 = detections[i, 1];
            float x2 = detections[i, 2];
            float y2 = detections[i, 3];
            int classId = (int)detections[i, 5];

            // Convert to x, y, width, height format
            float width = x2 - x1;
            float height = y2 - y1;

            boxes.Add(new BoundingBox
            {
                X = x1,
                Y = y1,
                Width = width,
                Height = height,
                Confidence = confidence,
                ClassIndex = classId
            });
        }

        return boxes;
    }
}