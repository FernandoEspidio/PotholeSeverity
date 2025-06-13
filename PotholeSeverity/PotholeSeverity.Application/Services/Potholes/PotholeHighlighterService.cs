using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PotholeSeverity.Application.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace PotholeSeverity.Application.Services.Potholes;

/// <summary>
/// Service for highlighting potholes in images by detecting bounding boxes using a pre-trained ML model.
/// </summary>
internal class PotholeHighlighterService : IPotholeHighlighterService
{
    private readonly ILogger<PotholeHighlighterService> _logger;
    private readonly InferenceSession _session;
    private const float ConfidenceThreshold = 0.5f;

    /// <summary>
    /// Initializes a new instance of the <see cref="PotholeHighlighterService"/> class.
    /// </summary>
    /// <param name="logger">The logger to use for logging.</param>
    public PotholeHighlighterService(ILogger<PotholeHighlighterService> logger)
    {
        _logger = logger;

        var modelPath = Path.Combine(AppContext.BaseDirectory, "Models", "PotholeHighlighter.onnx");
        _logger.LogInformation("Loading ONNX pothole highlighter model from: {ModelPath}", modelPath);
        _session = new InferenceSession(modelPath);
    }

    /// <inheritdoc />
    public Task<List<BoundingBox>> DetectPotholesAsync(byte[] imageBytes, CancellationToken cancellationToken = default)
    {
        try
        {
            using var image = Image.Load<Rgb24>(imageBytes);

            // Create a 3D tensor [channels, height, width]
            const int size = 640;
            image.Mutate(x => x.Resize(size, size));

            var tensor = new DenseTensor<float>(new[] { 3, size, size });

            // Copy the image data to the tensor
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        tensor[0, y, x] = pixelRow[x].R / 255.0f; // R channel
                        tensor[1, y, x] = pixelRow[x].G / 255.0f; // G channel
                        tensor[2, y, x] = pixelRow[x].B / 255.0f; // B channel
                    }
                }
            });

            // Run inference
            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

            using var results = _session.Run(inputs);

            // Get detections as a 2D tensor [num_detections, 6]
            // Each row: [x1, y1, x2, y2, confidence, class]
            var detections = results.First().AsTensor<float>();
            var boxes = new List<BoundingBox>();

            // Original dimensions for scaling if needed
            float imgWidth = image.Width;
            float imgHeight = image.Height;

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

            _logger.LogInformation("Detected {Count} potholes in image", boxes.Count);
            return Task.FromResult(boxes);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to detect potholes in image.");
            throw new InvalidOperationException("Pothole highlighting detection failed.", ex);
        }
    }
}
