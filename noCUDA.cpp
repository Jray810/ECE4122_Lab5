/**
 * Author: Raymond Jia
 * Course: ECE 4122
 * Professor: Dr. Hurley
 * Last Modified: 11/15/2021
 * 
 * Description: Calculating the steady state heat distribution on a metal plate.
 *              Solving Laplace's equation using the finite difference method.
 *              Implementation using OpenMP.
 * 
 * Instructions: To compile in Linux Terminal, run the command:
 *               $ g++ noCUDA.cpp -fopenmp -O3
**/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <omp.h>

#define ERROR 1

using namespace std;

void initializePreviousPlate(double* previousPlate, unsigned long N);
void initializeNextPlate(double* nextPlate, double* previousPlate, unsigned long N);
void updateNextPlate(double* previousPlate, double* nextPlate, unsigned long N);
void updatePreviousPlate(double* previousPlate, double* nextPlate, unsigned long N);
void printPlate(double* plate, unsigned long N);

int main(int argc, char *argv[])
{
    // Read command line inputs, check for five
    if (argc != 5)
    {
        cerr << "Invalid number of arguments! Expected 5, received " << argc << endl;
        return ERROR;
    }

    bool isN = false;
    unsigned long N;
    unsigned long I;

    // Command line input error checking
    string inputString = argv[1];
    if (inputString.length() != 2 || inputString[0] != '-' || !(inputString[1] == 'N' || inputString[1] == 'I'))
    {
        cerr << "Invalid second argument! Expected -N or -I, received " << inputString << endl;
        return ERROR;
    }
    else if (inputString[1] == 'N')
    {
        isN = true;
    }

    inputString = argv[2];
    for (int i=0; i<inputString.length(); ++i)
    {
        if (!isdigit(inputString[i]))
        {
            cerr << "Invalid value! Expected a number, received " << inputString << endl;
            return ERROR;
        }
    }
    if (isN)
    {
        N = stoul(inputString);
    }
    else
    {
        I = stoul(inputString);
    }

    inputString = argv[3];
    if (inputString.length() != 2 || inputString[0] != '-' || !(inputString[1] == 'N' || inputString[1] == 'I'))
    {
        cerr << "Invalid second argument! Expected -N or -I, received " << inputString << endl;
        return ERROR;
    }
    else if ((isN == true && inputString[1] == 'N') || (isN == false && inputString[1] == 'I'))
    {
        cerr << "Duplicated flag! Already received " << inputString << endl;
        return ERROR;
    }
    else if (inputString[1] == 'N')
    {
        isN = true;
    }
    else
    {
        isN = false;
    }

    inputString = argv[4];
    for (int i=0; i<inputString.length(); ++i)
    {
        if (!isdigit(inputString[i]))
        {
            cerr << "Invalid value! Expected a number, received " << inputString << endl;
            return ERROR;
        }
    }
    if (isN)
    {
        N = stoul(inputString);
    }
    else
    {
        I = stoul(inputString);
    }

    // Create plate arrays
    unsigned long fullSize = (N+2)*(N+2);
    double* previousPlate = new double[fullSize];
    double* nextPlate = new double[fullSize];

    // Initialize plate arrays
    initializePreviousPlate(previousPlate, N);
    initializeNextPlate(nextPlate, previousPlate, N);

    // Perform iterations
    for (int i=0; i<I; ++i)
    {
        updateNextPlate(previousPlate, nextPlate, N);
        updatePreviousPlate(previousPlate, nextPlate, N);
    }

    // Output temperature values
    printPlate(previousPlate, N+2);

    // Free allocated memory
    delete previousPlate;
    delete nextPlate;
}

void initializePreviousPlate(double* previousPlate, unsigned long N)
{
#pragma omp parallel shared(N, previousPlate)
    {
        unsigned long fullSize = (N+2)*(N+2);

#pragma omp for schedule(dynamic)
        for (int i=0; i<fullSize; ++i)
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
}

void initializeNextPlate(double* nextPlate, double* previousPlate, unsigned long N)
{
#pragma omp parallel shared(N, nextPlate, previousPlate)
    {
        unsigned long fullSize = (N+2)*(N+2);

#pragma omp for schedule(dynamic)
        for (int i=0; i<fullSize; ++i)
        {
            nextPlate[i] = previousPlate[i];
        }
    }
}

void updateNextPlate(double* previousPlate, double* nextPlate, unsigned long N)
{
#pragma omp parallel shared(N, previousPlate, nextPlate)
    {
        unsigned long fullSize = (N+2)*(N+2);

#pragma omp for schedule(dynamic)
        for (int i=0; i<fullSize; ++i)
        {
            if (!(i < (N+2) || i > (N+1)*(N+2) || i % (N+2) == 0 || (i+1) % (N+2) == 0))
            {
                nextPlate[i] = (previousPlate[i-N-2] +
                            previousPlate[i+N+2] +
                            previousPlate[i+1] +
                            previousPlate[i-1]) * 0.25;
            }
        }
    }
}

void updatePreviousPlate(double* previousPlate, double* nextPlate, unsigned long N)
{
#pragma omp parallel shared(N, previousPlate, nextPlate)
    {
        unsigned long fullSize = (N+2)*(N+2);

#pragma omp for schedule(dynamic)
        for (int i=0; i<fullSize; ++i)
        {
            previousPlate[i] = nextPlate[i];
        }
    }
}

void printPlate(double* plate, unsigned long N)
{
    ofstream fout ("finalTemperatures.csv", ios::out);

    unsigned long fullSize = (N+2)*(N+2);

    for (int i=0; i<N; ++i)
    {
        for (int j=0; j<N; ++j)
        {
            fout << std::fixed << setprecision(6) << plate[i*N + j] << ",";
        }
        fout << endl;
    }

    fout.close();
}