namespace PotholeSeverity.Application.Models
{
    /// <summary>
    /// Represents a bounding box for detected objects in an image.
    /// </summary>
    public class BoundingBox
    {
        /// <summary>
        /// Gets or sets the X coordinate of the bounding box.
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Gets or sets the Y coordinate of the bounding box.
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// Gets or sets the width of the bounding box.
        /// </summary>
        public float Width { get; set; }

        /// <summary>
        /// Gets or sets the height of the bounding box.
        /// </summary>
        public float Height { get; set; }

        /// <summary>
        /// Gets or sets the confidence score of the bounding box.
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// Gets or sets the class index of the detected object.
        /// </summary>
        public int ClassIndex { get; set; }
    }
}