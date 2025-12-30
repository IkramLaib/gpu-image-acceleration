# gpu-image-acceleration
Modern image processing pipelines often rely on GPU acceleration, but the benefits are highly dependent on workload size and memory transfer costs. This project aims to provide a clear, experimental comparison between CPU and GPU approaches, helping practitioners understand when GPU acceleration is truly advantageous.

# Performance Analysis: CPU vs GPU Image Processing

## Objective
The goal of this project is to evaluate the performance impact of GPU
acceleration for basic image processing operations compared to
CPU serial and CPU vectorized implementations.

## Operations
- Brightness adjustment: pixel + value
- Contrast enhancement: 128 + factor × (pixel − 128)

## Experimental Setup
- Image types: grayscale (camera), RGB (astronaut)
- Image sizes:
  - 512×512
  - 1024×1024
  - 2048×2048
- Hardware: NVIDIA GPU (cloud-based)
- GPU implementation: CUDA via Numba
- CPU implementations:
  - Serial nested loops
  - NumPy vectorized

## Methodology
For each image size and operation:
1. Execute CPU serial implementation
2. Execute CPU vectorized implementation
3. Execute GPU CUDA kernel
4. Measure wall-clock execution time
5. Repeat runs and record stable timings

GPU execution time includes kernel execution but excludes visualization.

## Results Summary
- CPU serial execution time increases quadratically with image size
- CPU vectorized implementation significantly outperforms serial code
- GPU implementation achieves the lowest execution time for large images
- For small images, GPU overhead dominates and CPU vectorization is faster

## Key Observations
- GPU acceleration is beneficial only beyond a certain data size
- Memory transfer and kernel launch overhead are non-negligible
- Parallelism efficiency increases with workload size

## Conclusion
GPU acceleration provides substantial speedups for large-scale image
processing workloads. However, optimized CPU vectorization remains
competitive for smaller images. Choosing the correct execution model
depends on workload scale and performance requirements.
