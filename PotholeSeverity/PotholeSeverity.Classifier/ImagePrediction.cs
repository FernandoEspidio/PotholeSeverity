using Microsoft.ML.Data;

namespace PotholeSeverity.Classifier;

/// <summary>
/// Prediction output (label + scores).
/// </summary>
public sealed class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = default!;

    public float[] Score { get; set; } = default!;
}
