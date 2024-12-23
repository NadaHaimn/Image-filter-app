# Image Filter Application - Built From Scratch

This project is a fully functional **Image Filter Application** created from scratch using **Python** and **Tkinter**. The application allows users to load, manipulate, and save images by applying a variety of advanced image filters and transformations. It provides a robust graphical user interface (GUI) for easy interaction and offers real-time visual feedback, making it ideal for anyone interested in image processing.

### Key Features
- **Load and Display Images**: Easily load `.jpg`, `.jpeg`, or `.png` image files into the application and preview them.
- **Apply Multiple Image Filters**: A wide array of filters and transformations are available to modify the image:
  - **Negative Filter**: Converts the image into its negative, inverting all pixel values.
  - **Power-Law Filter**: Adjusts the image intensity using a power law function, often used for enhancing or darkening images.
  - **Log Transformation**: Applies logarithmic transformation to the image, useful for images with a wide dynamic range.
  - **Gaussian Filter**: Applies a Gaussian blur to the image for smoothing and noise reduction.
  - **Histogram Matching**: Alters the image histogram to match the statistical properties of a reference image, improving contrast and brightness consistency.
  - **Edge Detection Filters**: Apply edge detection algorithms like **Sobel** and **Prewitt** to highlight edges in the image.
  - **Morphological Filters**: Implement **Average**, **Maximum**, **Minimum** and **Median** filters to modify pixel structures and remove noise.
  - **Real-Time Visualization**: All transformations are reflected immediately on the image displayed in the GUI, allowing users to see the effects instantly.
- **Save Image**: After applying the desired transformations, users can save the modified image in various formats.

### Technologies Used
- **Tkinter**: For building the GUI. Tkinter provides a simple, lightweight way to create graphical user interfaces in Python.
- **PIL (Pillow)**: Used for opening, manipulating, and saving images.
- **OpenCV**: A powerful library for advanced image processing tasks like edge detection, filtering, and transformations.
- **Numpy**: Used for matrix operations and efficient handling of pixel data in images.
- **Matplotlib**: Used to visualize and display image transformations and results in an interactive manner.

### Built From Scratch
This project demonstrates a strong understanding of image processing and GUI development, as it’s built entirely from scratch with no external frameworks or pre-built models. Here's why this project is a significant achievement:

1. **Custom Implementations**: All image manipulation algorithms, including edge detection, histograms, and filters like Gaussian and power-law, are custom-implemented using **OpenCV** and **Numpy**. This provides full control over the behavior of each transformation.
  
2. **Real-Time Interaction**: The application allows users to interact with the image and immediately see the effects of applied filters in real-time, all integrated within the same application without requiring complex setups or external software.

3. **Efficiency**: The application is optimized to work with standard image sizes and runs smoothly even on larger images, making it a practical tool for users working with high-resolution photos.

4. **User-Focused Design**: The GUI is designed with the end-user in mind, focusing on simplicity and ease of use. It requires no prior knowledge of image processing techniques to get started.

### Prerequisites

Before running the application, make sure you have **Python 3.x** installed on your machine along with the following libraries:

- `tkinter` (for GUI)
- `pillow` (for image manipulation)
- `opencv-python` (for advanced image processing)
- `numpy` (for efficient matrix handling)
- `matplotlib` (for visualizing images and transformations)

You can install the dependencies by running the following command:

```bash
pip install pillow opencv-python numpy matplotlib
