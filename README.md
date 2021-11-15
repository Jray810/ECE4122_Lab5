# ECE4122_Lab5

Lab5 Meta Thread
In this thread, we will clarify everything and provide a guide for you to start with. Please follow the specification closely and feel free to ask questions below.

[Revision & Clarification]

1. The plate has initial interior temperature 20°C and has fixed boundary temperature according to the figure in the instruction.


2. Input arguments are -N (interior dimension) and -I (UPPERCASE i, total iterations). Please ignore -q. [Note: N is interior dimension, so plate size should be N+2]


3. -N and -I both should take in a positive integer. Make sure you implement invalid/missing input handling. Please print:

Invalid parameters, please check your values.
in the console and terminate the program.

4. We will test -N up to 256 and -l up to 10000.

5. 40% of the top boundary area has 100°C and please use the condition: if n > 0.3*(N+2-1) && n < 0.7*(N+2-1), to define the area.

6. use double type for your array.

7. console output is cuda kernel execute time and it should be in milliseconds with 2 decimals. e.g.

10.01
and it should be under 1000ms. Here is a post about how to use cuda_event to measure kernel execute time: https://stackoverflow.com/questions/7876624/timing-cuda-operations

8. Please output overall plate (size N+2 * N+2) temperature in finalTemperatures.csv. Each temperature value should followed by a comma. Each row of temperatures has its own line. We will provide some test cases and expected results at the end of this post. Please set precision to at least 6 decimals.

[Coc-ice job & environment]

Create gpu job with vnc:

pace-vnc-job -l nodes=1:ppn=4:gpus=1 -l walltime=02:00:00 -q coc-ice-gpu

Create gpu job with command line only:

qsub -q coc-ice-gpu -A USERNAME -l nodes=1:ppn=4:gpus=1,walltime=02:00:00 -I

Load necessary module, compile and run by:

module load gcc/9.2.0 cuda/11.1

nvcc *.cu -o executable.out

./executable.out -N 100 -I 1000

[Guide]

1. You may first implement cpu version of the program. This can help you understand the problem setup, algorithm and you can also compare your solution with cuda version later.

2. https://github.com/NVIDIA/cuda-samples/blob/master/Samples/vectorAdd/vectorAdd.cu

This is a good example to start with. Basically the workflow is:
  1. allocate cpu memory for the array
  2. initialize value in the array
  3. allocate gpu memory for the array
  4. send cpu data to gpu
  5. kernel calculation
  6. send gpu data to cpu
  7. save result
  8. free memory


I suggest using 1d array instead of 2d array. See how indexing works in this case.

Use cudaGetDeviceProperties to get maximum threadperblock. Here is an example: https://cpp.hotexamples.com/examples/-/-/cudaGetDeviceProperties/cpp-cudagetdeviceproperties-function-examples.html

 

[Test cases]

1. -N 10 -I 100

case1.csv (precision to 6 decimals)

2. -N 100 -I 1000

case2.csv (precision to 6 decimals)
