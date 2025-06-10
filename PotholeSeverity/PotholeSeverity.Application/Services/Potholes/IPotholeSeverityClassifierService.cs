using PotholeSeverity.Application.Models.Potholes;

namespace PotholeSeverity.Application.Services.Potholes;

/// <summary>
/// Interface for classifying the severity of potholes based on images.
/// </summary>
public interface IPotholeSeverityClassifierService
{
    /// <summary>
    /// Classifies the severity of a pothole based on its image.
    /// </summary>
    /// <param name="image">The image of the pothole.</param>
    /// <param name="cancellationToken">A cancellation token to cancel the operation.</param>
    /// <returns>A <see cref="Task{Severity}"/> representing the asynchronous operation, with a <see cref="Severity"/> result indicating the severity of the pothole.</returns> 
    Task<Severity> ClassifyPotholeSeverityAsync(byte[] image, CancellationToken cancellationToken = default);
}
