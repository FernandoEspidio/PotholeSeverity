using Microsoft.ML.Data;

namespace PotholeSeverity.Application.Models;

/// <summary>
/// Prediction output (label + scores).
/// </summary>
public sealed class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = default!;

    public float[] Score { get; set; } = default!;
}
