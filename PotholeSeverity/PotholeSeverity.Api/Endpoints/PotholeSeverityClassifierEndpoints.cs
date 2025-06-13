using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.Extensions.ML;
using PotholeSeverity.Application;
using PotholeSeverity.Application.Models;
using PotholeSeverity.Application.Models.Potholes;
using PotholeSeverity.Application.Services.Potholes;
using static PotholeSeverity.Api.Models.PotholeSeverityClassifierEndpoints;

namespace PotholeSeverity.Api.Endpoints;

/// <summary>
/// Provides endpoints for pothole-related services including severity classification, detection, and highlighting.
/// </summary>
public static partial class PotholeSeverityClassifierEndpoints
{
    /// <summary>
    /// Adds endpoints for pothole-related services to the specified <see cref="IServiceCollection"/>.
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
        // Create a group for pothole-related endpoints
        var potholeEndpoints = app.MapGroup("/api/potholes");
        potholeEndpoints.WithTags("Pothole Services");

        // Endpoint to classify pothole severity based on an uploaded image file
        potholeEndpoints.MapPost("/classify", ClassifyPothole)
            .WithName("ClassifyPotholeSeverity")
            .Produces<Severity>(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status400BadRequest)
            .DisableAntiforgery();

        // Endpoint to detect pothole presence in an uploaded image
        potholeEndpoints.MapPost("/detect", DetectPothole)
            .WithName("DetectPotholePresence")
            .Produces<DetectionResult>(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status400BadRequest)
            .DisableAntiforgery();

        // Endpoint to highlight potholes (get bounding boxes) in an uploaded image
        potholeEndpoints.MapPost("/highlight", HighlightPotholes)
            .WithName("HighlightPotholes")
            .Produces<List<BoundingBox>>(StatusCodes.Status200OK)
            .Produces(StatusCodes.Status400BadRequest)
            .DisableAntiforgery();
    }

    public static async Task<IResult> ClassifyPothole(IPotholeSeverityClassifierService classifier, IFormFile file, CancellationToken cancellationToken)
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

    public static async Task<IResult> DetectPothole(IPotholeDetectorService detector, IFormFile file, CancellationToken cancellationToken)
    {
        if (file.Length == 0)
        {
            return Results.BadRequest("No file uploaded.");
        }

        using var memoryStream = new MemoryStream();
        await file.CopyToAsync(memoryStream, cancellationToken);
        var imageBytes = memoryStream.ToArray();

        var (index, label) = await detector.DetectPotholePresenceAsync(imageBytes, cancellationToken);
        return Results.Ok(new DetectionResult { Index = index, Label = label });
    }

    public static async Task<IResult> HighlightPotholes(IPotholeHighlighterService highlighter, IFormFile file, CancellationToken cancellationToken)
    {
        if (file.Length == 0)
        {
            return Results.BadRequest("No file uploaded.");
        }

        using var memoryStream = new MemoryStream();
        await file.CopyToAsync(memoryStream, cancellationToken);
        var imageBytes = memoryStream.ToArray();

        var boundingBoxes = await highlighter.DetectPotholesAsync(imageBytes, cancellationToken);
        return Results.Ok(boundingBoxes);
    }
}