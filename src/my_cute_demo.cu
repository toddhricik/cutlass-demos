#include <iostream>
#include <cutlass/cutlass.h>
//#include "cutlass/numeric/numeric_types.hpp"
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/util/print.hpp>

// Define a simple kernel to do a little work on a cuda device
__global__ void doSomething(cutlass::half_t x)
{
        // Print out the value of x * 2.0_hf
        printf("Device: %f\n", float(x * 2.0_hf));
        // Pause
	//for(int i=0; i < 1000000000; i++) { }
	//__syncthreads();
}
/*
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
	for (int m = 0; m < size<0>(layout); ++m) {
		for (int n = 0; n < size<1>(layout); ++n) {
			cute::print(layout(m,n));
		}
		printf("\n");
	}
}
*/
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

	using namespace cute;
	int M = 2048, N = 2048;
	float *d_S, *d_D;
	// Allocate and initialize d_S and d_D on device (omitted).
	// Create the row major layouts.
	auto tensor_shape = make_shape(M, N);
	auto tensor_shape_trans = make_shape(N, M);
	auto gmemLayoutS = make_layout(tensor_shape, GenRowMajor{});
	auto gmemLayoutD = make_layout(tensor_shape_trans, GenRowMajor{});

	// Create the row major tensors.
	Tensor tensor_S = make_tensor(make_gmem_ptr(d_S), gmemLayoutS);
	Tensor tensor_D = make_tensor(make_gmem_ptr(d_D), gmemLayoutD);

	// Create a column major layout. Note that we use (M,N) for shape.
	auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});

	// Create a column major view of the dst tensor.
	Tensor tensor_DT = make_tensor(make_gmem_ptr(d_D), gmemLayoutDT);

	Layout s8 = make_layout(Int<8>{});
	cute::print(s8);
	std::cout << std::endl;
	Layout d8 = make_layout(8);

	Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
	Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

	Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4), make_stride(Int<12>{},Int<1>{}));
	Layout s2xd4_col = make_layout(make_shape(Int<2>{},4), LayoutLeft{});
	Layout s2xd4_row = make_layout(make_shape(Int<2>{},4), LayoutRight{});

	Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)), make_stride(4,make_stride(2,1)));
        std::cout << std::endl;
	cute::print(d8);
	std::cout << std::endl;
	cute::print(s2xs4);
	std::cout << std::endl;
	// Return
        return 0;
}
