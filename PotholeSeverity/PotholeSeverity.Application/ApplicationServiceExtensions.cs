using Microsoft.Extensions.DependencyInjection;
using PotholeSeverity.Application.Services.Potholes;

namespace PotholeSeverity.Application;

public static class ApplicationServiceExtensions
{
    /// <summary>
    /// Adds application services to the specified <see cref="IServiceCollection"/>.
    /// </summary>
    /// <param name="services">The service collection to add services to.</param>
    public static void AddApplicationServices(this IServiceCollection services)
    {
        // Register services as scoped services
        services.AddScoped<IPotholeSeverityClassifierService, PotholeSeverityClassifierService>();
        services.AddScoped<IPotholeDetectorService, PotholeDetectorService>();
        services.AddScoped<IPotholeHighlighterService, PotholeHighlighterService>();
    }
}