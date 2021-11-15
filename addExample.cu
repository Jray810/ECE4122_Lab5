#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  cudaError_t err = cudaSuccess;

  int N = 1<<20;
  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  float *cudaX;
  float *cudaY;
  err = cudaMalloc(&cudaX, N*sizeof(float));
  if (err != cudaSuccess)
  {
      std::cout << "error 1\n" << cudaGetErrorString(err) << std::endl;
      return 1;
  }
  err = cudaMalloc(&cudaY, N*sizeof(float));
  if (err != cudaSuccess)
  {
      std::cout << "error 2\n" << cudaGetErrorString(err) << std::endl;
      return 1;
  }

  err = cudaMemcpy(cudaX, x, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
      std::cout << "error 3\n" << cudaGetErrorString(err) << std::endl;
      return 1;
  }
  err = cudaMemcpy(cudaY, y, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
      std::cout << "error 4\n" << cudaGetErrorString(err) << std::endl;
      return 1;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, cudaX, cudaY);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  err = cudaMemcpy(y, cudaY, N*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
      std::cout << "error 5\n" << cudaGetErrorString(err) << std::endl;
      return 1;
  }

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  delete x;
  delete y;

  // Free memory
  cudaFree(cudaX);
  cudaFree(cudaY);
  
  return 0;
}