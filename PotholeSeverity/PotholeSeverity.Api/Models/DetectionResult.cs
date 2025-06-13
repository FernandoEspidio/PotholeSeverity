namespace PotholeSeverity.Api.Models;

public static partial class PotholeSeverityClassifierEndpoints
{
    /// <summary>
    /// Represents the result of a pothole detection operation.
    /// </summary>
    public class DetectionResult
    {
        /// <summary>
        /// Gets or sets the index of the detected class.
        /// </summary>
        public int Index { get; set; }

        /// <summary>
        /// Gets or sets the label of the detected class.
        /// </summary>
        public string Label { get; set; } = string.Empty;
    }
}