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

// Main entrypoint
int main()
{
	using namespace cute;

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
	//Return
    return 0;
}
