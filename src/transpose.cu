#include <iostream>
#include <cutlass/cutlass.h>
//#include "cutlass/numeric/numeric_types.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/util/print.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
transposeKernelNaive(TensorS const S, TensorD const DT,
                ThreadLayoutS const tS, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  Tensor gDT = DT(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgDT = local_partition(gDT, tD, threadIdx.x);

  Tensor rmem = make_tensor_like(tSgS);

  copy(tSgS, rmem);
  copy(rmem, tDgDT);
}


// Main entrypoint
int main()
{
	using namespace cute;
	// This section is for construction of tensors
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
	// This section is for tiling tensors
	using bM = Int<64>;
	using bN = Int<64>;
 
	auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
	auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

	Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
	Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape_trans); // ((bN, bM), n', m')

	auto threadLayoutS =
		make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
	auto threadLayoutD =
		make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

	dim3 gridDim(
		size<1>(tiled_tensor_S),
		size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
	dim3 blockDim(size(threadLayoutS)); // 256 threads
	kernelTransposeNaive<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_DT, threadLayoutS, threadLayoutD);
	// Return
    return 0;
}
