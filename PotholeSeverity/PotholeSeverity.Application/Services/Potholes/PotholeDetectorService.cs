using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
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
/// Service for detecting the presence of potholes in images using a pre-trained ML model.
/// </summary>
internal class PotholeDetectorService : IPotholeDetectorService
{
    private readonly ILogger<PotholeDetectorService> _logger;
    private readonly InferenceSession _session;
    private readonly string[] _classes = { "normal", "pothole" };

    /// <summary>
    /// Initializes a new instance of the <see cref="PotholeDetectorService"/> class.
    /// </summary>
    /// <param name="logger">The logger to use for logging.</param>
    public PotholeDetectorService(ILogger<PotholeDetectorService> logger)
    {
        _logger = logger;

        var modelPath = Path.Combine(AppContext.BaseDirectory, "Models", "PotholeDetector.onnx");
        _logger.LogInformation("Loading ONNX pothole detector model from: {ModelPath}", modelPath);
        _session = new InferenceSession(modelPath);
    }

    /// <inheritdoc />
    public Task<(int index, string label)> DetectPotholePresenceAsync(byte[] imageBytes, CancellationToken cancellationToken = default)
    {
        try
        {
            // Process image and create tensor
            using var image = Image.Load<Rgb24>(imageBytes);
            image.Mutate(x => x.Resize(224, 224));

            var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

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

            // Run inference
            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            int predictedIndex = Array.IndexOf(output, output.Max());

            var label = _classes[predictedIndex];
            _logger.LogInformation("Predicted pothole presence: {Label}", label);

            return Task.FromResult((predictedIndex, label));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to detect pothole presence.");
            throw new InvalidOperationException("Pothole detection failed.", ex);
        }
    }
}
