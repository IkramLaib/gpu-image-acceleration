# Performance Analysis of Image Brightness and Contrast Enhancement
## CPU vs Vectorized vs GPU (CUDA)

## 1. Introduction
Image processing is a fundamental component of modern applications such as computer vision, medical imaging, autonomous systems, and multimedia platforms. Simple operations like brightness and contrast enhancement can become computationally expensive when applied to large images or datasets.

This project presents a performance comparison between three implementations:
- Serial CPU (nested loops)
- Vectorized CPU (NumPy)
- GPU implementation using CUDA

The goal is to evaluate execution time, scalability, and the practical benefit of GPU acceleration.

---

## 2. Objectives
- Implement brightness and contrast enhancement using explicit pixel-wise formulas  
- Use small and large images from multiple sources  
- Compare execution times across CPU and GPU implementations  
- Analyze scalability and performance trade-offs  

---

## 3. Dataset and Images

### 3.1 skimage Built-in Images
The following images from `skimage.data` were used:
- `camera` (grayscale)
- `astronaut` (RGB)

### 3.2 Generated Large Images
Synthetic images were generated to stress-test performance:
- Sizes from 512×512 to 4096×4096  
- Pixel values in range [0, 255]

---

## 4. Image Enhancement Methods

### 4.1 Brightness Adjustment
Formula:
```
new_pixel = pixel + value
```
All pixel values are clipped to [0, 255].

### 4.2 Contrast Adjustment
Formula:
```
new_pixel = 128 + factor * (pixel - 128)
```
All outputs are clipped to [0, 255].

---

## 5. Implementation Details

### 5.1 Serial CPU Implementation
- Uses nested loops
- Pixel-by-pixel computation
- High execution time for large images

### 5.2 Vectorized CPU Implementation
- Uses NumPy array operations
- Eliminates Python loops
- Significant speedup over serial CPU

### 5.3 GPU CUDA Implementation
- Implemented using CUDA kernels
- One thread per pixel
- Parallel execution on the GPU

Workflow:
1. Copy image to GPU
2. Launch CUDA kernel
3. Copy result back to CPU
4. Measure execution time

---

## 6. Experimental Setup
- Hardware: NVIDIA GPU (cloud-based)
- Software: Python, NumPy, scikit-image, Numba CUDA
- Timing tools: time.perf_counter, CUDA events

---

## 7. Results and Analysis

| Image Size | Serial CPU | Vectorized CPU | GPU CUDA |
|-----------|------------|----------------|----------|
| 512×512   | Slow       | Fast           | Moderate |
| 2048×2048 | Very Slow  | Moderate       | Fast     |
| 4096×4096 | Impractical| Slow           | Very Fast|

Key observations:
- Serial CPU does not scale
- Vectorization provides strong gains
- GPU excels for large workloads

---

## 8. Visualization
- Original vs enhanced images were displayed
- Brightness and contrast effects are clearly visible
- CPU and GPU outputs match numerically

---

## 9. Discussion
This study shows that:
- Vectorization should be attempted before GPU usage
- GPU overhead is non-negligible for small images
- CUDA provides the best performance for large-scale processing

---

## 10. Conclusion
GPU acceleration using CUDA significantly outperforms CPU-based implementations for large images. While vectorized CPU solutions are efficient and simple, GPUs remain the optimal solution for high-throughput image processing.

---

## 11. Future Work
- Shared memory optimization
- Multi-GPU processing
- Batch image pipelines
- Real-time applications

---

## 12. References
- NVIDIA CUDA Programming Guide  
- scikit-image Documentation  
- NumPy Performance Guide  

