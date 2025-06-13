namespace PotholeSeverity.Classifier;

public static class Program
{
    private const string SampleImage = "sample-image.png";

    public static void Main()
    {
        Console.WriteLine($"Running predictions on {SampleImage}\n");

        // ---- Severity Classification ----
        var (sevIdx, sevLabel) = SeverityClassifier.Predict(SampleImage);
        Console.WriteLine($"Severity prediction : {sevLabel} (index {sevIdx})\n");

        // ---- Binary Detection ----
        var (detIdx, detLabel) = PotholeDetector.Predict(SampleImage);
        Console.WriteLine($"Pothole presence    : {detLabel} (index {detIdx})\n");

        // ---- Object Detection (Bounding Boxes) ----
        var boxes = PotholeHighlighter.Predict(SampleImage);
        Console.WriteLine("Bounding boxes detected:");
        foreach (var box in boxes)
        {
            Console.WriteLine($"  X={box.X}, Y={box.Y}, W={box.Width}, H={box.Height}, Conf={box.Confidence:F2}");
        }
        if (boxes.Count == 0)
        {
            Console.WriteLine("  No potholes detected.");
        }
    }
}