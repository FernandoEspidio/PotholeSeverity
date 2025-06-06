using System.ComponentModel;

namespace PotholeSeverity.Application.Models.Potholes;

/// <summary>
/// Represents the severity of a pothole.
/// </summary>
public enum Severity
{
    /// <summary>
    /// Indicates an unknown severity level, typically used when the severity has not been assessed or is not applicable.
    /// </summary>
    [Description("unknown_pothole")]
    Unknown = -1,
    
    /// <summary>
    /// Indicates a minor pothole that does not significantly affect driving conditions.
    /// </summary>
    [Description("minor_pothole")]
    Minor = 0,

    /// <summary>
    /// Indicates a moderate pothole that may cause some discomfort or minor damage to vehicles.
    /// </summary>
    [Description("medium_pothole")]
    Moderate = 1,

    /// <summary>
    /// Indicates a severe pothole that poses a significant risk to vehicles and drivers.
    /// </summary>
    [Description("major_pothole")]
    Severe = 2
}