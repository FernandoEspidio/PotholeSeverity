using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PotholeSeverity.Classifier
{
    public static class SeverityClassifier
    {
        private const string ModelPath = "PotholeSeverityModel.onnx";
        private static readonly string[] Classes = { "minor_pothole", "medium_pothole", "major_pothole" };

        /// <summary>
        /// Predicts the severity class index and label for the given image.
        /// </summary>
        public static (int index, string label) Predict(string imagePath)
        {
            DenseTensor<float> inputTensor = ImageUtils.LoadAsTensor(imagePath);

            using var session = new InferenceSession(ModelPath);
            string inputName = session.InputMetadata.Keys.First();

            var results = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            });

            float[] logits = results.First().AsEnumerable<float>().ToArray();
            int predIdx = Array.IndexOf(logits, logits.Max());
            return (predIdx, Classes[predIdx]);
        }
    }
}