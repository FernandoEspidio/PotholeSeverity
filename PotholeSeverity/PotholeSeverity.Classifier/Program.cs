namespace PotholeSeverity.Classifier;

using System.Drawing;
using System.Drawing.Imaging;
using PotholeSeverity.Application.Models;

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

        // Draw bounding boxes on the image and save as new file
        DrawBoundingBoxes(SampleImage, boxes);
    }

    private static void DrawBoundingBoxes(string imagePath, List<BoundingBox> boxes)
    {
        try
        {
            if (boxes.Count == 0)
            {
                Console.WriteLine("No boxes to draw.");
                return;
            }

            // Generate output filename
            string outputFilename = Path.GetFileNameWithoutExtension(imagePath) + "_highlighted" + Path.GetExtension(imagePath);

            // Use SkiaSharp for cross-platform image processing
            using (var inputStream = File.OpenRead(imagePath))
            using (var skBitmap = SkiaSharp.SKBitmap.Decode(inputStream))
            using (var skImage = SkiaSharp.SKImage.FromBitmap(skBitmap))
            using (var skSurface = SkiaSharp.SKSurface.Create(new SkiaSharp.SKImageInfo(skBitmap.Width, skBitmap.Height)))
            {
                var canvas = skSurface.Canvas;
                canvas.DrawImage(skImage, 0, 0);

                using (var paint = new SkiaSharp.SKPaint
                {
                    Color = SkiaSharp.SKColors.Red,
                    StrokeWidth = 3,
                    IsStroke = true
                })
                {
                    foreach (var box in boxes)
                    {
                        var rect = new SkiaSharp.SKRect(box.X, box.Y, box.X + box.Width, box.Y + box.Height);
                        canvas.DrawRect(rect, paint);
                    }
                }

                using (var output = File.OpenWrite(outputFilename))
                using (var img = skSurface.Snapshot())
                using (var data = img.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100))
                {
                    data.SaveTo(output);
                }
                Console.WriteLine($"\nHighlighted image saved as: {outputFilename}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error drawing bounding boxes: {ex.Message}");
        }
    }
}