namespace PotholeSeverity.Application.Models;

/// <summary>
/// Represents one image/label row.
/// </summary>
public sealed class ImageData
{
    public string ImagePath { get; set; } = default!;
    public string Label { get; set; } = default!;
}
