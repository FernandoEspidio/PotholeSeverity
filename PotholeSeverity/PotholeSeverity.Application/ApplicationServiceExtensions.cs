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
        // Register the PotholeSeverityClassifier as a scoped service
        services.AddScoped<IPotholeSeverityClassifierService, PotholeSeverityClassifierService>();
    }
}