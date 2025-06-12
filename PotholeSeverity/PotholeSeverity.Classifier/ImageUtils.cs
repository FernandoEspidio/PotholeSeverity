using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PotholeSeverity.Classifier
{
    public static class ImageUtils
    {
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        /// <summary>
        /// Loads an image from <paramref name="path"/> and returns a normalised DenseTensor<float>
        /// with shape [1,3,224,224] (NCHW) suitable for ResNet‑style ONNX models.
        /// </summary>
        public static DenseTensor<float> LoadAsTensor(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Image not found: {path}");

            using var image = Image.Load<Rgb24>(path);
            image.Mutate(ctx => ctx.Resize(224, 224));

            var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    Rgb24 pixel = image[x, y];
                    tensor[0, 0, y, x] = ((pixel.R / 255f) - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = ((pixel.G / 255f) - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = ((pixel.B / 255f) - Mean[2]) / Std[2];
                }
            }
            return tensor;
        }
    }
}