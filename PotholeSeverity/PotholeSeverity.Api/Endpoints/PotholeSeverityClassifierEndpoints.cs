using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.Extensions.ML;
using PotholeSeverity.Application;
using PotholeSeverity.Application.Models;
using PotholeSeverity.Application.Models.Potholes;
using PotholeSeverity.Application.Services.Potholes;

namespace PotholeSeverity.Api.Endpoints;

/// <summary>
/// Provides endpoints for classifying the severity of potholes.
/// </summary>
public static class PotholeSeverityClassifierEndpoints
{
    /// <summary>
    /// Adds endpoints for classifying pothole severity to the specified <see cref="IServiceCollection"/>.
    /// </summary>
    /// <param name="services">The service collection to add endpoints to.</param>
    public static void AddPotholeSeverityClassifierEndpoints(this IServiceCollection services)
    {
        services.AddPredictionEnginePool<ImageData, ImagePrediction>()
            .FromFile(Path.Combine(AppContext.BaseDirectory, "PotholeSeverityModel.zip"));

        services.AddApplicationServices();
    }

    public static void MapPotholeSeverityClassifierEndpoints(this IEndpointRouteBuilder app)
    {
        // Create a group for pothole severity classifier endpoints
        var potholeSeverityClassifier = app.MapGroup("/pothole-severity-classifier");
        potholeSeverityClassifier.WithTags("Pothole Severity Classifier");

        // Endpoint to classify pothole severity based on an uploaded image file
        potholeSeverityClassifier.MapPost("/api/potholes/classify", ClassifyPothole)
        .WithName("ClassifyPotholeSeverity")
        .Produces<Severity>(StatusCodes.Status200OK)
        .Produces(StatusCodes.Status400BadRequest)
        .DisableAntiforgery();
    }

    public static async Task<IResult> ClassifyPothole(IPotholeSeverityClassifier classifier, IFormFile file, CancellationToken cancellationToken)
    {
        if (file.Length == 0)
        {
            return Results.BadRequest("No file uploaded.");
        }

        using var memoryStream = new MemoryStream();
        await file.CopyToAsync(memoryStream, cancellationToken);
        var imageBytes = memoryStream.ToArray();

        var severity = await classifier.ClassifyPotholeSeverityAsync(imageBytes, cancellationToken);
        return Results.Ok(severity);
    }
}