# Image Processing Project

## Overview
This project focuses on implementing various image processing techniques using Python and OpenCV. It includes scripts for Fourier transform, image filtering (convolution, averaging, Gaussian, and median filters), as well as contrast and brightness adjustment. The project aims to demonstrate how these techniques can be applied to enhance and manipulate digital images.

## Files

1. `fourier_filtering.py`: Contains functions for Fourier transform, Gaussian filter smoothing in the Fourier domain, and saving the transformed images.

2. `image_filtering.py`: Includes functions for image convolution, averaging filter, Gaussian filter, and median filter. It also performs contrast and brightness adjustment on images.

3. `Noisy_image.png`: Input grayscale image used for various image processing tasks.

4. `Uexposed.png`: Input image for contrast and brightness adjustment.

5. `adjusted_image.png`: Output image after contrast and brightness adjustment.

6. `average_image.png`: Output image after applying an averaging filter.

7. `convolved_image.png`: Output image after convolution.

8. `gaussian_image.png`: Output image after applying a Gaussian filter.

9. `guassian_fourier.png`: Output image after Gaussian filter smoothing in the Fourier domain.

10. `converted_fourier.png`: Output image after converting the input image to the Fourier domain.

11. `median_image.png`: Output image after applying a median filter.

## Instructions

1. Ensure you have Python 3 installed along with the required libraries (`numpy`, `opencv-python`).

2. Clone this repository to your local machine.

3. Run the scripts using Python:
   - For Fourier transform and Gaussian filter smoothing: `python fourier_filtering.py`
   - For image filtering tasks: `python image_filtering.py`

4. The output images will be saved in the same directory with the respective names mentioned above.

## Additional Notes

- Adjust parameters in the scripts as needed for different image processing tasks.
- Experiment with different filter sizes and values for optimal results.
