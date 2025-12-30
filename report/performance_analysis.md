#Performance Analysis of Image Brightness and Contrast Enhancement
#CPU vs Vectorized vs GPU (CUDA)
#1. Introduction
Image processing is a core component of many modern applications, including computer vision, medical imaging, autonomous systems, and multimedia. Operations such as brightness and contrast enhancement are computationally simple but become expensive when applied to large images or datasets.

The objective of this project is to analyze and compare the performance of three implementations of brightness and contrast enhancement:

Serial CPU implementation (nested loops)

Vectorized CPU implementation (NumPy)

GPU implementation using CUDA

The study focuses on execution time, scalability, and performance gains when leveraging GPU acceleration.

#2. Objectives
The main goals of this project are:

Implement brightness and contrast enhancement using explicit pixel-wise formulas

Use both small and large images from different sources

Compare execution times across CPU and GPU implementations

Analyze scalability and performance trade-offs

Demonstrate practical benefits of CUDA acceleration

#3. Dataset and Images
#3.1 skimage Built-in Images
Two image types from skimage.data were used:

camera (grayscale, medium resolution)

astronaut (RGB, higher complexity)

#3.2 Large Generated Images
To stress-test performance, synthetic images were generated:

Sizes ranging from 512×512 to 4096×4096

Pixel values in range [0, 255]

This combination allows evaluation on both realistic and large-scale workloads.

#4. Image Enhancement Methods
#4.1 Brightness Adjustment
Brightness enhancement is defined as:

ini
Copy code
new_pixel = pixel + value
Where:

value ∈ ℝ controls brightness increase or decrease

After computation, values are clipped to [0, 255].

#4.2 Contrast Adjustment
Contrast enhancement is defined as:

ini
Copy code
new_pixel = 128 + factor × (pixel − 128)
Where:

factor > 1 increases contrast

0 < factor < 1 reduces contrast

All outputs are clipped to [0, 255].

5. Implementation Details
5.1 Serial CPU Implementation
Implemented using nested for-loops

Pixel-by-pixel computation

Serves as a baseline reference

Computationally expensive for large images

Characteristics:

Simple and readable

Poor scalability

High execution time

5.2 Vectorized CPU Implementation
Implemented using NumPy array operations

Eliminates explicit Python loops

Uses optimized low-level routines

Characteristics:

Much faster than serial CPU

Still limited by CPU memory bandwidth

Easy to implement and maintain

5.3 GPU CUDA Implementation
Implemented using CUDA kernels

One thread per pixel

Massive parallel execution

Two kernels were written:

Brightness adjustment kernel

Contrast adjustment kernel

Workflow:

Copy image from host to device

Launch CUDA kernel

Copy result back to host

Measure kernel execution time

Characteristics:

Excellent scalability

Significant speedup for large images

Overhead noticeable for small images

6. Experimental Setup
Hardware: NVIDIA GPU (cloud-based)

Software:

Python

NumPy

scikit-image

Numba CUDA

Matplotlib

Timing:

CPU: time.perf_counter()

GPU: CUDA events for accurate measurement

Each experiment was repeated multiple times and averaged.

7. Experiments and Results
7.1 Brightness and Contrast Parameters
Tested values:

Brightness: +30, +60, -30

Contrast factors: 0.8, 1.2, 1.5

7.2 Image Size Comparison
Image Size	Serial CPU	Vectorized CPU	GPU CUDA
512×512	Slow	Fast	Moderate
2048×2048	Very Slow	Moderate	Fast
4096×4096	Impractical	Slow	Very Fast

7.3 Performance Analysis
Serial CPU execution time grows linearly with pixel count

Vectorized CPU offers significant speedup (~10–30×)

GPU CUDA provides the highest performance for large images

GPU overhead dominates for small images

8. Visualization
Original vs enhanced images were visualized

Brightness and contrast effects are clearly observable

GPU and CPU outputs are numerically identical (within precision)

9. Discussion
This study highlights several key insights:

Naive serial implementations are unsuitable for large-scale image processing

Vectorization is an essential optimization step before GPU usage

GPU acceleration excels when workload size is sufficiently large

CUDA programming introduces complexity but yields substantial gains

10. Conclusion
The project demonstrates that GPU acceleration using CUDA significantly outperforms CPU-based implementations for image brightness and contrast enhancement, especially for large images.

While vectorized CPU implementations provide a strong middle ground, GPUs remain the optimal solution for high-throughput image processing pipelines.

11. Future Work
Possible extensions include:

Use of shared memory for improved CUDA performance

Multi-GPU processing

Batch processing of image datasets

Integration into real-time computer vision pipelines

12. References
NVIDIA CUDA Programming Guide

scikit-image Documentation

NumPy Performance Optimization Guide
