# Gaussian and Laplacian Pyramids, Image Blending, and Mosaicking

This project implements core multiresolution image processing techniques using Gaussian and Laplacian pyramids, including image reduction, expansion, reconstruction, and blending. It also performs seamless image mosaicking using pyramid-based blending.

## Overview

The project is modularized into Python scripts, each handling a key stage in the multiresolution pipeline:
- reduce.py: Downsamples an image using Gaussian filtering
- expand.py: Upsamples an image with interpolation and smoothing
- gaussianPyramid.py: Constructs a Gaussian pyramid from an image
- laplacianPyramid.py: Builds a Laplacian pyramid by subtracting Gaussian levels
- reconstruct.py: Reconstructs the original image from a Laplacian pyramid
- convolve.py: Custom convolution operation (used internally)
- mosaic.py: Performs image blending across a seam using Laplacian pyramids

## Folder Contents

- images/: Contains input images for blending and mosaicking
- *_pyramid_level_*.jpg: Intermediate outputs from Gaussian and Laplacian pyramids
- *_image.jpg: Processed outputs from convolve, reduce, expand, and reconstruct stages

## Examples

- A1andA2.jpg, B1andB2.jpg, etc.: Input image pairs for blending
- reconstructed_image.jpg: Output of reconstructing an image from its Laplacian pyramid
- expanded_image.jpg, reduced_image.jpg: Outputs after applying expand/reduce
- convolved_image_x.jpg: Result of convolving the image with a kernel

## Requirements

- Python 3.7+
- OpenCV
- NumPy

Install dependencies with:
pip install opencv-python numpy

## How to Run

You can run each module individually for specific functionality:
python reduce.py
python expand.py
python gaussianPyramid.py
python laplacianPyramid.py
python reconstruct.py
python mosaic.py

Ensure that images are located in the images/ folder or hardcoded paths in the scripts are updated accordingly.

## Key Concepts

- Gaussian Pyramid: A set of progressively blurred and downsampled versions of the image
- Laplacian Pyramid: Difference between levels of the Gaussian pyramid, useful for edge preservation
- Reconstruction: Uses the Laplacian pyramid and successive expansion to recover the original image
- Blending/Mosaicking: Combines image halves at different pyramid levels for seamless transitions


