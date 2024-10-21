#include <iostream>
#include <cutlass/cutlass.h>
//#include "cutlass/numeric/numeric_types.hpp"
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cuda_runtime.h>
#include <stdio.h>
// Define a simple kernel to do a little work on a cuda device
__global__ void doSomething(cutlass::half_t x)
{
        // Print out the value of x * 2.0_hf
        printf("Device: %f\n", float(x * 2.0_hf));
        // Pause
	//for(int i=0; i < 1000000000; i++) { }
	//__syncthreads();
}

// Main entrypoint
int main()
{
        // Declare and instantiate some cutlass type half_t variables
        cutlass::half_t x = 0.5_hf;
        cutlass::half_t y = 0.0_hf;

//	cudaMalloc(void** 
        // Read in a value from stdin and save the input to x
        std::cin >> x;
        // Write the Host side data for x to stdout
        std::cout << "Host: " << 2.0_hf * x << std::endl;
        // Hop into a device kernel parameterized by x
        doSomething<<< dim3(1,1,1), dim3(1,1,1) >>>(x);
	cudaDeviceSynchronize();
	for(int i=0; i<10000000; i++)
	{
		// pass
	}
        // Synchronize to allow printf inside kernel function to access the propper stream
//	if(cudaErr != cudaSuccess){
//		std::cout << "problem13 " << cudaErr << std::endl;
//	}
	// Return
        return 0;
}
