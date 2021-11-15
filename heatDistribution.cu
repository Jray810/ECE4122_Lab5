/**
 * Author: Raymond Jia
 * Course: ECE 4122
 * Professor: Dr. Hurley
 * Last Modified: 11/15/2021
 * 
 * Description: This program calculates the steady state heat distribution after a given number of iterations
 *              of a thin metal plate with provided inner dimensions. It solves Laplace's equation using
 *              the finite difference method. The program takes advantage of CUDA multithreading to quickly
 *              perform the necessary calculations and outputs the final temperature values to a csv file. It
 *              also prints the time spent in the CUDA kernel to the terminal in ms.
 * 
 * Instructions: To compile:
 *               $ nvcc heatDistribution.cu -o heatDistro.out
 *               To run:
 *               $ nvprof ./heatDistro.out -N <dimensions> -I <iterations>
**/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#define ERROR 1

__global__ void initializePreviousPlate(double* previousPlate, unsigned long N);
__global__ void initializeNextPlate(double* nextPlate, double* previousPlate, unsigned long N);
__global__ void updateNextPlate(double* previousPlate, double* nextPlate, unsigned long N);
__global__ void updatePreviousPlate(double* previousPlate, double* nextPlate, unsigned long N);
void printPlate(double* plate, unsigned long N);
bool cudaErrorCheck(cudaError_t error);

using namespace std;

/**
 * Main
 * Inputs: Command Line inputs
 * Outputs: Kernel time to terminal
 *          Final temperatures to finalTemperatures.csv file
 * Description: Handles overarching operations, timing of CUDA kernel, and passing
 *              of memory between CPU and GPU
 **/
int main(int argc, char *argv[])
{
    // Read command line inputs, check for five
    if (argc != 5)
    {
        cerr << "Invalid number of arguments! Expected 5, received " << argc << endl;
        return ERROR;
    }

    // Boolean to determine which flag
    bool isN = false;

    // Flag values
    unsigned long N;    // Inner dimension
    unsigned long I;    // Iterations

    // ------------- Check for valid first flag -------------
    string inputString = argv[1];
    if (inputString.length() != 2 || inputString[0] != '-' || !(inputString[1] == 'N' || inputString[1] == 'I'))
    {
        cerr << "Invalid parameters, please check your values." << endl;
        return ERROR;
    }
    else if (inputString[1] == 'N')
    {
        // Set isN to true if first flag is '-N' otherwise false for first flag is '-I'
        isN = true;
    }

    // ------------- Check for valid first flag value -------------
    inputString = argv[2];
    for (int i=0; i<inputString.length(); ++i)
    {
        // Make sure it is a whole number
        if (!isdigit(inputString[i]))
        {
            cerr << "Invalid parameters, please check your values." << endl;
            return ERROR;
        }
    }
    // Make sure the value is nonzero
    if (stoul(inputString) == 0)
    {
        cerr << "Invalid parameters, please check your values." << endl;
        return ERROR;
    }
    // Set the correct flag value
    if (isN)
    {
        N = stoul(inputString);
    }
    else
    {
        I = stoul(inputString);
    }

    // ------------- Check for valid second flag -------------
    inputString = argv[3];
    if (inputString.length() != 2 || inputString[0] != '-' || !(inputString[1] == 'N' || inputString[1] == 'I'))
    {
        cerr << "Invalid parameters, please check your values." << endl;
        return ERROR;
    }
    else if ((isN == true && inputString[1] == 'N') || (isN == false && inputString[1] == 'I'))
    {
        // Make sure it is not a duplicate flag
        cerr << "Duplicated flag! Already received " << inputString << endl;
        return ERROR;
    }
    else if (inputString[1] == 'N')
    {
        // Set isN to true if second flag is '-N'
        isN = true;
    }
    else
    {
        // Set isN to false if second flag is '-I'
        isN = false;
    }

    // ------------- Check for valid second flag value -------------
    inputString = argv[4];
    for (int i=0; i<inputString.length(); ++i)
    {
        // Make sure it is a whole number
        if (!isdigit(inputString[i]))
        {
            cerr << "Invalid parameters, please check your values." << endl;
            return ERROR;
        }
    }
    // Make sure the value is nonzero
    if (stoul(inputString) == 0)
    {
        cerr << "Invalid parameters, please check your values." << endl;
        return ERROR;
    }
    // Set the correct flag value
    if (isN)
    {
        N = stoul(inputString);
    }
    else
    {
        I = stoul(inputString);
    }

    // Set full plate array size
    unsigned long fullSize = (N+2)*(N+2);

    // Allocate CPU memory for plate array
    double* plate = new double[fullSize];

    // Create GPU plate array pointers
    double* cudaPP;     // Previous Plate
    double* cudaNP;     // Next Plate
    
    // Allocate GPU memory for Previous Plate
    if (!cudaErrorCheck(cudaMalloc(&cudaPP, fullSize*sizeof(double))))
    {
        return ERROR;
    }
    
    // Allocate GPU memory for Next Plate
    if (!cudaErrorCheck(cudaMalloc(&cudaNP, fullSize*sizeof(double))))
    {
        return ERROR;
    }

    // CUDA event objects for timing
    cudaEvent_t start, stop;

    // Create CUDA start event
    if (!cudaErrorCheck(cudaEventCreate(&start)))
    {
        return ERROR;
    }

    // Create CUDA stop event
    if (!cudaErrorCheck(cudaEventCreate(&stop)))
    {
        return ERROR;
    }

    // Timestamp the CUDA start event
    if (!cudaErrorCheck(cudaEventRecord(start, 0)))
    {
        return ERROR;
    }

    // Initialize previous plate array using CUDA kernel
    initializePreviousPlate<<<N+2, N+2>>>(cudaPP, N);
    cudaDeviceSynchronize();

    // Initialize next plate array using CUDA kernel
    initializeNextPlate<<<N+2, N+2>>>(cudaNP, cudaPP, N);
    cudaDeviceSynchronize();

    // Perform desired number of iterations
    for (int i=0; i<I; ++i)
    {
        // Update next plate array using CUDA kernel
        updateNextPlate<<<N+2, N+2>>>(cudaPP, cudaNP, N);
        cudaDeviceSynchronize();

        // Update previous plate array from next plate array using CUDA kernel
        updatePreviousPlate<<<N+2, N+2>>>(cudaPP, cudaNP, N);
        cudaDeviceSynchronize();
    }

    // Copy plate memory from GPU to CPU
    if (!cudaErrorCheck(cudaMemcpy(plate, cudaPP, fullSize*sizeof(double), cudaMemcpyDeviceToHost)))
    {
        return ERROR;
    }

    // Timestamp the CUDA stop event
    if (!cudaErrorCheck(cudaEventRecord(stop)))
    {
        return ERROR;
    }

    // Synchronize
    if (!cudaErrorCheck(cudaEventSynchronize(stop)))
    {
        return ERROR;
    }

    // Calculate elapsed time in ms
    float elapsedTime;
    if (!cudaErrorCheck(cudaEventElapsedTime(&elapsedTime, start, stop)))
    {
        return ERROR;
    }

    // Print elapsed time in ms to terminal
    cout << fixed << setprecision(2) << elapsedTime << endl;

    // Destroy the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Output temperature values to csv file
    printPlate(plate, N+2);

    // Free allocated GPU memory
    cudaFree(cudaPP);
    cudaFree(cudaNP);

    // Free allocated CPU memory
    delete plate;

    return 0;
}

/**
 * initializePreviousPlate
 * Inputs: double* previousPlate - Pointer to previous plate state
 *         unsigned long N - Inner dimensions of grid
 * Outputs: None
 * Description: Initializes values of previous plate, all values are 20 degrees C
 *              except for specific section on top edge that has value of 100 degrees C
 **/
__global__ void initializePreviousPlate(double* previousPlate, unsigned long N)
{
    // Set the stride of loop based on thread parameters
    int stride = blockDim.x * gridDim.x;

    // Calculate full size of the plate
    unsigned long fullSize = (N+2)*(N+2);

    // Loop through elements of the plate and set values accordingly
    for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<fullSize; i+=stride)
    {
        if (i > (N+1)*0.3 && i < (N+1)*0.7)
        {
            previousPlate[i] = 100.0;
        }
        else
        {
            previousPlate[i] = 20.0;
        }
    }
}

/**
 * initializeNextPlate
 * Inputs: double* nextPlate - Pointer to next plate state
 *         double* previousPlate - Pointer to previous plate state
 *         unsigned long N - Inner dimensions of grid
 * Outputs: None
 * Description: Copies previous plate values to next plate values.
 **/
__global__ void initializeNextPlate(double* nextPlate, double* previousPlate, unsigned long N)
{
    // Set the stride of loop based on thread parameters
    int stride = blockDim.x * gridDim.x;

    // Calculate the full size of the plate
    unsigned long fullSize = (N+2)*(N+2);

    // Loop through elements of next plate and set equal to elements of previous plate
    for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<fullSize; i+=stride)
    {
        nextPlate[i] = previousPlate[i];
    }
}

/**
 * updateNextPlate
 * Inputs: double* previousPlate - Pointer to previous plate state
 *         double* nextPlate - Pointer to next plate state
 *         unsigned long N - Inner dimensions of grid
 * Outputs: None
 * Description: Updates elements of next plate based off of the elements of previous plate
 *              using the finite difference method. (Average of 4 neighboring points)
 **/
__global__ void updateNextPlate(double* previousPlate, double* nextPlate, unsigned long N)
{
    // Set the stride of loop based on thread parameters
    int stride = blockDim.x * gridDim.x;

    // Calculate the full size of the plate
    unsigned long fullSize = (N+2)*(N+2);

    // Loop through elements of next plate and set equal to the average of its
        // 4 neighboring points extracted from previous plate
    for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<fullSize; i+=stride)
    {
        // Only update points that are not on the boundary
        if (!(i < (N+2) || i > (N+1)*(N+2) || i % (N+2) == 0 || (i+1) % (N+2) == 0))
        {
            // Take the average of neighboring points
            nextPlate[i] = (previousPlate[i-N-2] +
                        previousPlate[i+N+2] +
                        previousPlate[i+1] +
                        previousPlate[i-1]) * 0.25;
        }
    }
}

/**
 * updatePreviousPlate
 * Inputs: double* previousPlate - Pointer to previous plate state
 *         double* nextPlate - Pointer to next plate state
 *         unsigned long N - Inner dimensions of grid
 * Outputs: None
 * Description: Copies next plate values to previous plate values.
 **/
__global__ void updatePreviousPlate(double* previousPlate, double* nextPlate, unsigned long N)
{
    // Set the stide of loop based on thread parameters
    int stride = blockDim.x * gridDim.x;

    // Calculate the full size of the plate
    unsigned long fullSize = (N+2)*(N+2);

    // Loop through elements of previous plate and set equal to elements of next plate
    for (int i=blockIdx.x * blockDim.x + threadIdx.x; i<fullSize; i+=stride)
    {
        previousPlate[i] = nextPlate[i];
    }
}

/**
 * printPlate
 * Inputs: double* plate - Pointer to the current plate state
 *         unsigned long N - Dimensions of grid
 * Outputs: None
 * Description: Outputs comma separated plate state values to finalTemperatures.csv
 **/
void printPlate(double* plate, unsigned long N)
{
    // Create output file stream object
    ofstream fout ("finalTemperatures.csv", ios::out);

    // Iterate through the rows of the plate and print comma separated values to output file
    for (int i=0; i<N; ++i)
    {
        for (int j=0; j<N; ++j)
        {
            fout << std::fixed << setprecision(6) << plate[i*N + j] << ",";
        }
        fout << endl;
    }

    // Close output file stream object
    fout.close();
}

/**
 * cudaErrorCheck
 * Input: cudaError_t error - CUDA error object
 * Outputs: bool - True if there is no error, False if an error occurred
 * Description: Checks if the provided error object indicated an error.
 *              If an error occurred, print information to terminal and return
 *              false to notify the caller. Otherwise return true to indicate
 *              that no error occurred.
 **/
bool cudaErrorCheck(cudaError_t error)
{
    // Check if error object is cudaSuccess
    if (error != cudaSuccess)
    {
        // Print error information to terminal
        cout << "CUDA Error Occurred.\n" << cudaGetErrorString(error) << endl;
        return false;
    }

    // No error, return true
    return true;
}