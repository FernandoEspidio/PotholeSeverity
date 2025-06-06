using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Mvc;
using PotholeSeverity.Api.Endpoints;
using PotholeSeverity.Application.Models.Potholes;
using Serilog;

var builder = WebApplication.CreateSlimBuilder(args);

builder.Services.Configure<RouteOptions>(options =>
{
    options.ConstraintMap["regex"] = typeof(Microsoft.AspNetCore.Routing.Constraints.RegexRouteConstraint);
});

builder.Services.Configure<JsonOptions>(options =>
{
    options.JsonSerializerOptions.TypeInfoResolverChain.Insert(0, JsonContext.Default);
});

Log.Logger = new LoggerConfiguration()
    .WriteTo.Console()
    .WriteTo.File("../Logs/log-.txt", rollingInterval: RollingInterval.Day)
    .Enrich.FromLogContext()
    .MinimumLevel.Information()
    .CreateLogger();

builder.Host.UseSerilog();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(options =>
{
    options.SwaggerDoc("v1", new() { Title = "My API", Version = "v1" });
});

builder.Services.AddPotholeSeverityClassifierEndpoints();

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();

app.MapPotholeSeverityClassifierEndpoints();

app.Run();

[JsonSerializable(typeof(Severity))]
public partial class JsonContext : JsonSerializerContext { }
