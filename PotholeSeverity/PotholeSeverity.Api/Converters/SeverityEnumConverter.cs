using System.ComponentModel;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using PotholeSeverity.Application.Models.Potholes;

public class SeverityEnumConverter : JsonConverter<Severity>
{
    public override Severity Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        var input = reader.GetString();
        foreach (Severity value in Enum.GetValues(typeof(Severity)))
        {
            var desc = value.GetType()
                            .GetField(value.ToString())?
                            .GetCustomAttribute<DescriptionAttribute>()?.Description;
            if (string.Equals(desc, input, StringComparison.OrdinalIgnoreCase))
            {
                return value;
            }
        }

        throw new JsonException($"Invalid value '{input}' for Severity.");
    }

    public override void Write(Utf8JsonWriter writer, Severity value, JsonSerializerOptions options)
    {
        var desc = value.GetType()
                        .GetField(value.ToString())?
                        .GetCustomAttribute<DescriptionAttribute>()?.Description
                   ?? value.ToString();

        writer.WriteStringValue(desc);
    }
}
