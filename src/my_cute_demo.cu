#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

// Define a simple kernel to do a little work on a cuda device
__global__ void kernel(cutlass::half_t x)
{
        // Print out the value of x * 2.0_hf
        printf("Device: %f\n", float(x * 2.0_hf));
        // Pause
        for(int i=0; i<100000; i++) {  }
}

// Main entrypoint
int main()
{
        // Declare and instantiate some cutlass type half_t variables
        cutlass::half_t x = 0.5_hf;
        cutlass::half_t y = 0.0_hf;
        // Read in a value from stdin and save the input to x
        std::cin >> x;
        // Write the Host side data for x to stdout
        std::cout << "Host: " << 2.0_hf * x << std::endl;
        // Hop into a device kernel parameterized by x
        kernel<<< dim3(1,1,1), dim3(1,1,1) >>> (x);
        // Synchronize to allow printf inside kernel function to access the propper stream
        cudaDeviceSynchronize();
        // Return
        return 0;
}
