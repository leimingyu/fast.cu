#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <vector>
#include <random>
#include <cassert>

#include <iomanip>  // setfill, setw
#include <cstdint>  // int8_t/uint8_t
#include <cstdarg>
#include <cstdio>



#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
// #include <cuda_bf16.h>

#define DEBUG 1
#define K32 32 
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// typedef __nv_bfloat16 bf16;


//----------------------------------------------------------------------------//
// Utility 
//----------------------------------------------------------------------------//
void logMessage(const char* format, ...) {
    // Get the current timestamp
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    // Print the timestamp
    std::cout << "[" << timestamp << "] ";

    // Handle the variadic arguments
    va_list args;
    va_start(args, format);
    vprintf(format, args);  // Print the formatted message
    va_end(args);

    // End with a newline
    std::cout << std::endl;
}


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

//----------------------------------------------------------------------------//
// Functions 
//----------------------------------------------------------------------------//
template <int TILE_K>
void runTest(std::vector<uint8_t> current_test_ab,
             uint16_t current_test_c,
             std::vector<uint32_t> &current_result);





//----------------------------------------------------------------------------//
// wgmma used in "examples/matmul/matmul_2.cuh"   (bf16)
//----------------------------------------------------------------------------//
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ---------------------------------------------------------------------
// Helpers to build SMEM descriptors with no swizzling
// ---------------------------------------------------------------------
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
  return ((x & 0x3FFFFULL) >> 4);
}

__device__ uint64_t make_smem_desc(uint8_t *base_ptr, int ld_major, int ld_minor)
{
  // Convert pointer to SMEM address
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(base_ptr));
  uint64_t desc = 0ULL;

  // Encode base address bits
  desc |= matrix_descriptor_encode(addr);

  // Encode leading dimensions (ld_major, ld_minor)
  desc |= (matrix_descriptor_encode((uint64_t)ld_major) << 16);  // leading dimentin byte offset
  desc |= (matrix_descriptor_encode((uint64_t)ld_minor) << 32);  // stide dimension byte offset

  // Turn off the 128B swizzle => do *not* set bit 62
  // desc |= (1ULL << 62);  // <--- commented out for no swizzling

  return desc;
}

// ---------------------------------------------------------------------
// WGMMA fence/commit/wait
// ---------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void wgmma_wait_group() {
  static_assert(N >= 0 && N <= 7, "wgmma_wait_group: N must be in [0..7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}




// template <typename Tin, int BlockMajorSize, int BlockMinorSize>
// void create_tensor_map(CUtensorMap *tma_map, Tin* gmem_ptr, int blocks_height, int blocks_width) {
//     void* gmem_address = (void*)gmem_ptr;
//     uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1};
//     uint64_t gmem_prob_stride[5] = {sizeof(Tin), sizeof(Tin) * BlockMinorSize*blocks_width, 0, 0, 0};

//     uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
//     uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

//     CUresult result = cuTensorMapEncodeTiled(
//         tma_map,
// 		// CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
// 		CU_TENSOR_MAP_DATA_TYPE_UINT8,
// 		2, gmem_address, gmem_prob_shape,
//         gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
// 		CU_TENSOR_MAP_INTERLEAVE_NONE,
//         // CU_TENSOR_MAP_SWIZZLE_128B,
// 		CU_TENSOR_MAP_SWIZZLE_NONE,
// 		CU_TENSOR_MAP_L2_PROMOTION_NONE,
// 		CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

//     assert(result == CUDA_SUCCESS);
// }



// template <typename Tin ,int BlockMajorSize, int BlockMinorSize>
// __host__ static inline CUtensorMap* allocate_and_create_tensor_map(Tin* src, int blocks_height, int blocks_width) {
// 	logMessage("%s", __func__);
// 	logMessage("blocks height = %d", blocks_height);
// 	logMessage("blocks width  = %d", blocks_width);
	
//     CUtensorMap *tma_map_d;
//     cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
//     CUtensorMap tma_map_src;
//     create_tensor_map<Tin, BlockMajorSize, BlockMinorSize>(&tma_map_src, src, blocks_height, blocks_width);
//     cudaMemcpy(tma_map_d, &tma_map_src, sizeof(CUtensorMap), cudaMemcpyHostToDevice);  //  ?  d2d?
//     return tma_map_d;
// }


//////////////////////////////////////////////////////////////////////////////
// Minimal inline PTX for wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e5m2
// storing two 32-bit accumulators (c0, c1).
//////////////////////////////////////////////////////////////////////////////

// ref:  https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp#L19019C1-L19052C3

template<int scaleD, int scaleA, int scaleB>
__device__ __forceinline__
void wgmma_m64n8k32_f16_e5m2_e5m2(
    uint64_t descA,
    uint64_t descB,
    uint32_t &c0,
    uint32_t &c1)
{
  // scaleD is turned into a predicate (p) in PTX:
  // if scaleD != 0 => accumulate; else => overwrite
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  setp.ne.b32 p, %4, 0;\n"
    "  wgmma.mma_async.sync.aligned.m64n8k32.f16.e5m2.e5m2 "
    "  { %0, %1 }, %2, %3, p, %5, %6;\n"
    "}\n"
    : "+r"(c0), "+r"(c1)
    : "l"(descA), "l"(descB),
      "n"((int32_t)scaleD),
      "n"((int32_t)scaleA),
      "n"((int32_t)scaleB)
  );
}

// ---------------------------------------------------------------------
// Kernel:	single block of 128 threads. We do a single 64x8x32 MMA
// 			from FP8 E5M2 => accum in FP16, stored in two 32-bit regs.
// ---------------------------------------------------------------------
__global__ void matmul_fp8e5m2_64x8x32_kernel(
	const uint8_t *A, // [64*32] = 2048 bytes
	const uint8_t *B, // [32*8 ] = 256  bytes
	uint32_t *C)	
{
	// We'll copy entire A and B into SMEM. (No TMA, no advanced cp.async.)
	__shared__ uint8_t sA[64 * 32]; // 2048 bytes
	__shared__ uint8_t sB[32 * 8];  // 256 bytes

	// We'll do naive copy from global to shared:
	int tid = threadIdx.x;
	// Copy A (2048 elements)
	for (int i = tid; i < 64 * 32; i += blockDim.x)
	{
		sA[i] = A[i];
	}
	// Copy B (256 elements)
	for (int i = tid; i < 32 * 8; i += blockDim.x)
	{
		sB[i] = B[i];
	}
	__syncthreads();

    if(tid == 0) {
        printf("\nsA\n");
        for (int i = 0; i < 64 * 32; i++)
        {
            if((i % 32) == 0) printf("\n");
            printf("%2X ", sA[i]);
        }
        printf("\n");

        printf("\nsB\n");
        for (int i = 0; i < 32 * 8; i++)
        {
            if((i % 8) == 0) printf("\n");
            printf("%2X ", sB[i]);
        }
        printf("\n");
    }

	// Build SMEM descriptors for A/B. No swizzle.
	// For a 64x32 chunk: each row is 32 columns => ld_major=32, arbitrary ld_minor=1024
	//uint64_t descA = make_smem_desc(sA, /*ld_major=*/32, /*ld_minor=*/1024);
	uint64_t descA = make_smem_desc(sA, /*ld_major=*/32, /*ld_minor=*/64);

	// For a 32x8 chunk: each row is 8 columns => ld_major=8, arbitrary ld_minor=512
	// uint64_t descB = make_smem_desc(sB, /*ld_major=*/8, /*ld_minor=*/512);
	uint64_t descB = make_smem_desc(sB, /*ld_major=*/8, /*ld_minor=*/32);

	// Our accumulators: 2 x 32-bit registers => 4 total fp16 values
	// C is 64x8 , each warp read 16x8 of inputC, there are 32 threads per warp, so each fiber hold 4 input values.
	// ideally, we would like fiber0 holds the 1st 2 inputs, the offset by 8x8 elements, hold the 2nd 2 inputs.

	// here, fiber reads the 1st 2xfp16, and next 2xfp16
	// noted that, we only care about the c0 input 
	uint32_t c0 = C[2*tid + 0];
	uint32_t c1 = C[2*tid + 1];

	// Start the WGMMA group
	wgmma_fence();

	// One wgmma call for the entire 64x8x32
	// scaleD=1 => accumulate with existing data in c0,c1
	// scaleA=1 => unscaled input A
	// scaleB=1 => unscaled input B
	wgmma_m64n8k32_f16_e5m2_e5m2<1,1,1>(descA, descB, c0, c1);

	// Commit, then wait for group 0
	wgmma_commit_group();
	wgmma_wait_group<0>();

	// Now c0,c1 each hold two FP16 accumulators. In a real code,
	// you'd map lane IDs carefully to store each portion of the 64x8 tile.
	// For demonstration, we simply store c0,c1 from each thread to global mem.

	int store_idx = tid; // 0..127
	C[2 * store_idx + 0] = c0;
	C[2 * store_idx + 1] = c1;

	// print the lower half of c0
	uint16_t c0_lo = static_cast<uint16_t>(c0 & 0xFFFF);
    uint16_t c0_hi = static_cast<uint16_t>((c0 >> 16) & 0xFFFF);
    // uint16_t c1_lo = static_cast<uint16_t>(c1 & 0xFFFF);
    // uint16_t c1_hi = static_cast<uint16_t>((c1 >> 16) & 0xFFFF);

	if(tid == 0) {
		// printf("[tid=%d] c0_lo=0x%04X c0_hi=0x%04X  c1_lo=0x%04X c1_hi=0x%04X\n", tid, c0_lo, c0_hi, c1_lo, c1_hi);
		printf("[tid=%d] c0_lo=0x%4X c0_hi=0x%4X  \n", tid, c0_lo, c0_hi);
		// printf("[tid=%d] c0_lo=0x%4X \n", tid, c0_lo);
		// printf("[tid=%d] \n", tid);
	}
}

//----------------------------------------------------------------------------//
// Main 
//----------------------------------------------------------------------------//
int main(int argc, char **argv)
{
	//------------------------------------------------------------------------//
	// Read commandlines
	//------------------------------------------------------------------------//
	if (argc != 2)
	{
		std::cerr << "\nUsage: " << argv[0] << " <filename>" << std::endl;
		return 1;
	}

	//------------------------------------------------------------------------//
	// Read all test cases
	//------------------------------------------------------------------------//
	// 65 input values per row:   c +  32 of a/b
	std::vector<uint16_t> allTests_c;			   // input c in f16
	std::vector<std::vector<uint8_t>> allTests_ab; // input a/b in fp8

	std::cout << "file : " << argv[1] << std::endl;

	std::ifstream file(argv[1], std::ifstream::ate | std::ifstream::binary);
	if (!file.is_open())
	{
		std::cerr << "Unable to open file " << argv[1] << std::endl;
		return 1;
	}

	// Get the size of the file
	std::streamsize fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	std::cout << "Read all the tests cases ... " << std::endl;
	std::string line;
	while (getline(file, line))
	{
		// read line :    c a0 b0 a1 b1 ....  aN bN
		std::istringstream iss(line);
		std::vector<uint8_t> numbers_ab; // current line for a/b
		std::string hexStr;

		// read c first
		iss >> hexStr;
		uint16_t num_c = static_cast<uint16_t>(std::stoul(hexStr, nullptr, 16));
		allTests_c.push_back(num_c); // store current line for C

		// read a/b
		while (iss >> hexStr)
		{
			uint8_t num = static_cast<uint8_t>(std::stoul(hexStr, nullptr, 16));
			numbers_ab.push_back(num);
		}

		allTests_ab.push_back(numbers_ab); // store current line for a and b
	}

	std::cout << std::endl;

	file.close();

	//------------------------------------------------------------------------//
    // Check first line : c + 32x{a(i), b(i)}
    //------------------------------------------------------------------------//
#if DEBUG
    printf("\nCheck first line of input file:\n");

    printf("%04X ", allTests_c[0]);

    for (int i = 0; i < 64; i++)
    {
        printf("%02X ", allTests_ab[0][i]);
    }
    printf("\n\n");
#endif

    //------------------------------------------------------------------------//
    // Run all test cases
    //------------------------------------------------------------------------//
    std::cout << "\nRun Tensor Core Tests\n" << std::endl;

    // prepare results
    int totalNum = static_cast<int>(allTests_ab.size());

	// results in 32 results?
    std::vector<std::vector<uint32_t>> allTests_results(totalNum);

	for (int i = 0; i < totalNum; i++)
	{
		logMessage("case : %d\n", i);

		// each test inputs
		uint16_t current_test_c = allTests_c[i];
		std::vector<uint8_t> current_test_ab = allTests_ab[i];

		// output
		std::vector<uint32_t> current_result;

		//--------------------------------------------------------------------//
		// run tensor core test
		//--------------------------------------------------------------------//
		runTest<K32>(current_test_ab, current_test_c, current_result);

		allTests_results[i] = current_result;
	}

/*
      for (int kernel_num : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) {
        // for (int kernel_num : {0, 11}) {
        // Give the GPU some rest to avoid thermal throttling
        sleep(5);
        std::cout << "KERNEL " << kernel_num << std::endl;
        // Verify against cuBLAS. Also serves as a warmup step.
        if (run_verif) {
          memset(C, 0, sizeof(bf16) * max_size * max_size);
          cudaCheck(cudaMemcpy(dC, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
          cudaCheck(cudaMemcpy(dC_ref, C, sizeof(bf16) * max_size * max_size, cudaMemcpyHostToDevice));
          memset(DB, ~0, sizeof(int) * max_size * 128);
          cudaCheck(cudaMemcpy(dDB, DB, sizeof(int) * max_size * 128,
            cudaMemcpyHostToDevice));
          run_kernel(0, m, n, k, dA, dB, dC_ref); // cuBLAS
          run_kernel(kernel_num, m, n, k, dA, dB, dC, dDB); // Executes the kernel, modifies the result matrix
          cudaCheck(cudaDeviceSynchronize());
          cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
          cudaMemcpy(C, dC, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);
          cudaMemcpy(C_ref, dC_ref, sizeof(bf16) * max_size * max_size, cudaMemcpyDeviceToHost);

          if (kernel_num > 1 && !verify_matrix(C_ref, C, m * n)) {
            std::cout << "~~~~~~~~~~~~~~~~ Failed to pass the correctness verification against cuBLAS. ~~~~~~~~~~~~~~~~" << std::endl;
            printf("%f\n", __bfloat162float(C_ref[m]));
          }

          cudaMemcpy(DB, dDB, sizeof(int) * max_size * 8, cudaMemcpyDeviceToHost);

      */
    return 0;
};

//----------------------------------------------------------------------------//
// host code to prepare test
//----------------------------------------------------------------------------//
template <int TILE_K>
void runTest(std::vector<uint8_t> current_test_ab,
			 uint16_t current_test_c,
			 std::vector<uint32_t> &current_result)
{
	//  C in f16 , A and B in FP8

	//------------------------------------------------------------------------//
	// total workload size M 64 x N 8 x K 32
	//------------------------------------------------------------------------//
	long max_size = 64;
	long M = 64, N = 8, K = 32;

	//------------------------------------------------------------------------//
	// prepare host buffers
	//------------------------------------------------------------------------//
	uint8_t *hA = nullptr;
	uint8_t *hB = nullptr;

	uint32_t *hD = nullptr;
	uint32_t *hresult = nullptr; 

	size_t sizeA = M * K;
	size_t sizeB = N * K;
	size_t sizeCD = M * N / 2;   // noted: for fp16 accumulation, each 32 reg will store two fp16 results

	hA = (uint8_t *)malloc(sizeof(uint8_t) * sizeA);
	hB = (uint8_t *)malloc(sizeof(uint8_t) * sizeB);
	hD = (uint32_t *)malloc(sizeof(uint32_t) * sizeCD);
	hresult = (uint32_t *)malloc(sizeof(uint32_t) * sizeCD);

	// init to 0
	memset(hA, 0, sizeof(uint8_t) * sizeA);
	memset(hB, 0, sizeof(uint8_t) * sizeB);
	memset(hD, 0, sizeof(uint32_t) * sizeCD);
	memset(hresult, 0, sizeof(uint32_t) * sizeCD);

	//------------------------------------------------------------------------//
	// read/set up data on cpu
	//------------------------------------------------------------------------//
	std::cout << "Read inputs a/b " << std::endl;
	for (int i = 0; i < TILE_K; i++)
	{
        // MxK
		hA[i] = current_test_ab[i * 2];		//  read a

        // KxN
		uint8_t val_b = current_test_ab[i * 2 + 1]; //  read b
        hB[i * N] =  val_b;
	}





	std::cout << "Read input C" << std::endl;
	// pack fp16 into a 32 bit register
	uint32_t packed = 0;
	packed |= (uint32_t)current_test_c & 0xFFFF;    // put low half 
	// packed |= ((uint32_t)half_hi & 0xFFFF) << 16;    // put high half 
	hD[0] = packed; 

    printf("packed into 32bit : %08X \n\n", hD[0]);



	//------------------------------------------------------------------------//
	// gpu buffer
	//------------------------------------------------------------------------//
	uint8_t *dA = nullptr;
	uint8_t *dB = nullptr;
	uint32_t *dD = nullptr;

	cudaMalloc((void **)&dA, sizeof(uint8_t) * sizeA);
	cudaMalloc((void **)&dB, sizeof(uint8_t) * sizeB);
	cudaMalloc((void **)&dD, sizeof(uint32_t) * sizeCD);

	// h2d : copy input ops to gpu
	cudaMemcpy(dA, hA, sizeof(uint8_t) * sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(uint8_t) * sizeB, cudaMemcpyHostToDevice);
	cudaMemcpy(dD, hD, sizeof(uint32_t) * sizeCD, cudaMemcpyHostToDevice);

	// Launch a single block with 128 threads => "1 warpgroup" in your test
    matmul_fp8e5m2_64x8x32_kernel<<<1, 128>>>(dA, dB, dD);

	// d2h : copy results back to host
	cudaMemcpy(hresult, dD, sizeof(uint32_t) * sizeCD, cudaMemcpyDeviceToHost);

// check value
#if DEBUG
	uint32_t c0 = hresult[0];
	uint16_t c0_lo = static_cast<uint16_t>(c0 & 0xFFFF);
	uint16_t c0_hi = static_cast<uint16_t>((c0 >> 16) & 0xFFFF);
	printf("[tid=0] c0_lo=0x%04X c0_hi=0x%04X \n", c0_lo, c0_hi);
	printf("%08X\n", hresult[0]);
#endif

	// current_result.push_back(result_cpu[0]);
	// current_result.push_back(result_cpu[1]);


	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dD);

	free(hA);
	free(hB);
	free(hD);
	free(hresult);

	cudaDeviceSynchronize();
}