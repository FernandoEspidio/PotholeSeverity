using System.Threading;
using System.Threading.Tasks;

namespace PotholeSeverity.Application.Services.Potholes;

/// <summary>
/// Interface for detecting the presence of potholes in images.
/// </summary>
public interface IPotholeDetectorService
{
    /// <summary>
    /// Detects whether a pothole is present in an image.
    /// </summary>
    /// <param name="image">The image to analyze for pothole presence.</param>
    /// <param name="cancellationToken">A cancellation token to cancel the operation.</param>
    /// <returns>A <see cref="Task{ValueTuple{int, string}}"/> representing the asynchronous operation, with a tuple containing the detection index and label.</returns>
    Task<(int index, string label)> DetectPotholePresenceAsync(byte[] image, CancellationToken cancellationToken = default);
}
