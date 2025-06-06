using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using PotholeSeverity.Application.Models;
using PotholeSeverity.Application.Models.Potholes;

namespace PotholeSeverity.Application.Services.Potholes;

/// <summary>
/// Classifies the severity of potholes based on images using a pre-trained ML model.
/// </summary>
internal class PotholeSeverityClassifier : IPotholeSeverityClassifier
{
    private readonly ILogger<PotholeSeverityClassifier> _logger;
    private readonly PredictionEnginePool<ImageData, ImagePrediction> _predictionEnginePool;

    /// <summary>
    /// Initializes a new instance of the <see cref="PotholeSeverityClassifier"/> class.
    /// </summary>
    /// <param name="logger">The logger to use for logging.</param>
    /// <param name="predictionEnginePool">The ML.NET prediction engine pool.</param>
    public PotholeSeverityClassifier(
        ILogger<PotholeSeverityClassifier> logger,
        PredictionEnginePool<ImageData, ImagePrediction> predictionEnginePool)
    {
        _logger = logger;
        _predictionEnginePool = predictionEnginePool;
    }

    /// <inheritdoc />
    public async Task<Severity> ClassifyPotholeSeverityAsync(byte[] imageBytes, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Classifying pothole severity for image of size {ImageSize} bytes", imageBytes.Length);

        var tempFilePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.jpg");

        try
        {
            _logger.LogInformation("Writing image bytes to temporary file: {TempFilePath}", tempFilePath);
            await File.WriteAllBytesAsync(tempFilePath, imageBytes, cancellationToken);

            var imageData = new ImageData { ImagePath = tempFilePath };

            _logger.LogInformation("Predicting pothole severity for image at {TempFilePath}", tempFilePath);

            var prediction = _predictionEnginePool.Predict(imageData);

            _logger.LogInformation("Prediction completed with label: {PredictedLabel}", prediction.PredictedLabel);

            return prediction.PredictedLabel switch
            {
                "minor_pothole" => Severity.Minor,
                "medium_pothole" => Severity.Moderate,
                "major_pothole" => Severity.Severe,
                _ => Severity.Unknown
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error classifying pothole severity for image at {TempFilePath}", tempFilePath);
            throw new InvalidOperationException("Failed to classify pothole severity.", ex);
        }
        finally
        {
            if (File.Exists(tempFilePath))
            {
                _logger.LogInformation("Deleting temporary file: {TempFilePath}", tempFilePath);
                File.Delete(tempFilePath);
            }
        }
    }
}
