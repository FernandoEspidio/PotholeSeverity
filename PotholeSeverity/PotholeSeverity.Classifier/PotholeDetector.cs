using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PotholeSeverity.Classifier
{
    public static class PotholeDetector
    {
        private const string ModelPath = "PotholeDetector.onnx"; // output of train_pothole_detector.py
        private static readonly string[] Classes = { "normal", "pothole" };

        /// <summary>
        /// Returns detection result (index, label) for pothole presence.
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