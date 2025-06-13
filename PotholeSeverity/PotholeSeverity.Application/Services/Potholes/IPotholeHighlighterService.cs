using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using PotholeSeverity.Application.Models;

namespace PotholeSeverity.Application.Services.Potholes;

/// <summary>
/// Interface for highlighting potholes in images by detecting bounding boxes.
/// </summary>
public interface IPotholeHighlighterService
{
    /// <summary>
    /// Detects potholes in an image and returns their bounding boxes.
    /// </summary>
    /// <param name="image">The image to analyze for potholes.</param>
    /// <param name="cancellationToken">A cancellation token to cancel the operation.</param>
    /// <returns>A <see cref="Task{List{BoundingBox}}"/> representing the asynchronous operation, with a list of bounding boxes for the detected potholes.</returns>
    Task<List<BoundingBox>> DetectPotholesAsync(byte[] image, CancellationToken cancellationToken = default);
}
