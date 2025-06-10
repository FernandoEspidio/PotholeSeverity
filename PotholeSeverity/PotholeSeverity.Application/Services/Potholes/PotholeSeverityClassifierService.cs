using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PotholeSeverity.Application.Models.Potholes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PotholeSeverity.Application.Services.Potholes;

/// <summary>
/// Classifies the severity of potholes based on images using a pre-trained ML model.
/// </summary>
internal class PotholeSeverityClassifierService : IPotholeSeverityClassifierService
{
    private readonly ILogger<PotholeSeverityClassifierService> _logger;
    private readonly InferenceSession _session;
    private readonly string[] _classes = { "minor_pothole", "medium_pothole", "major_pothole" };

    /// <summary>
    /// Initializes a new instance of the <see cref="PotholeSeverityClassifierService"/> class.
    /// </summary>
    /// <param name="logger">The logger to use for logging.</param>
    /// <param name="predictionEnginePool">The ML.NET prediction engine pool.</param>
    public PotholeSeverityClassifierService(
        ILogger<PotholeSeverityClassifierService> logger)
    {

        _logger = logger;

        var modelPath = Path.Combine(AppContext.BaseDirectory, "Models", "PotholeSeverityModel.onnx");
        _logger.LogInformation("Loading ONNX model from: {ModelPath}", modelPath);
        _session = new InferenceSession(modelPath);
    }

    /// <inheritdoc />
    public Task<Severity> ClassifyPotholeSeverityAsync(byte[] imageBytes, CancellationToken cancellationToken = default)
    {
        try
        {
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

            var inputName = _session.InputMetadata.Keys.First();
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            int predictedIndex = Array.IndexOf(output, output.Max());

            var label = _classes[predictedIndex];
            _logger.LogInformation("Predicted label: {Label}", label);

            var severity = label switch
            {
                "minor_pothole" => Severity.Minor,
                "medium_pothole" => Severity.Moderate,
                "major_pothole" => Severity.Severe,
                _ => Severity.Unknown
            };

            _logger.LogInformation("Classified severity: {Severity}", severity);
            return Task.FromResult(severity);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to classify pothole severity.");
            throw new InvalidOperationException("Prediction failed.", ex);
        }
    }
}
