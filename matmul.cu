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


#define DEBUG 1
#define K32 32 

// Add an enum for FP8 format type
enum class FP8Format {
    E5M2,
    E4M3
};


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
template <int TILE_K, FP8Format FORMAT>
void runTest(std::vector<uint8_t> current_test_ab,
             uint16_t current_test_c,
             std::vector<uint16_t> &current_result);


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
  uint64_t desc = 0ULL; // default to 0 

  // Encode base address bits
  desc |= matrix_descriptor_encode(addr);
  // Encode leading dimensions (ld_major, ld_minor)
  desc |= (matrix_descriptor_encode((uint64_t)ld_major) << 16);  // leading dimentin byte offset
  desc |= (matrix_descriptor_encode((uint64_t)ld_minor) << 32);  // stide dimension byte offset

// 	// no swi
//   desc |= (((uint64_t)ld_major) << 16);  // leading dimentin byte offset
//   desc |= (((uint64_t)ld_minor) << 32);  // stide dimension byte offset

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

// ref: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm90_gmma.hpp#L12984-L13022

template<int scaleD, int scaleA, int scaleB>
__device__ __forceinline__
void wgmma_m64n8k32_f16_e4m3_e4m3(
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
      "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3 "
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
// Modify the kernel to be templated
template<FP8Format FORMAT>
__global__ void matmul_fp8_64x8x32_kernel(
	const uint8_t *A, // [64*32] = 2048 bytes
	const uint8_t *B, // [32*8 ] = 256  bytes
	uint32_t *C)	
{
	// We'll copy entire A and B into SMEM. (No TMA, no advanced cp.async.)
	//__shared__ uint8_t sA[64 * 32]; // 2048 bytes
	//__shared__ uint8_t sB[32 * 8];  // 256 bytes

	__shared__ alignas(128) uint8_t sA[64 * 64];
	__shared__ alignas(128) uint8_t sB[64 * 64];

	// We'll do naive copy from global to shared:
	int tid = threadIdx.x;

	// set shared memory to 0
	for (int i = tid; i < 64 * 32; i += blockDim.x)
	{
		// sA[i] = A[i];
		sA[i] = 0;
	}

	for (int i = tid; i < 32 * 8; i += blockDim.x)
	{
		// sB[i] = B[i];
		sB[i] = 0; 
	}
	__syncthreads();

	// read only the first 32 elements of A and B
	if(tid < 32) {
		//sA[tid * 64] = A[tid];
		//sB[tid*8] = B[tid]; // KxN

		//// try1: A is MxK, B is NxK
		sA[tid] = A[tid];
		sB[tid] = B[tid];

		//// try2: A is MxK, B is KxN
		//sA[tid] = A[tid];
		//sB[tid*8] = B[tid];

		// try3: A is KxN, B is KxN
		// sA[tid*64] = A[tid];
		// sB[tid*8] = B[tid];
	}	
	__syncthreads();


	/*
	if(tid == 0) 
	{
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
			if((i % 32) == 0) printf("\n");
			printf("%2X ", sB[i]);
		}
		printf("\n");
	}
	*/

	// Build SMEM descriptors for A/B. No swizzle.

	//-----------//
	// try1 :
	//-----------//
	uint64_t descA = make_smem_desc(&sA[0], 16, 1024);
	uint64_t descB = make_smem_desc(&sB[0], 16, 1024);

	// 16
	// uint64_t descA = make_smem_desc(sA, /*ld_major=*/32, /*ld_minor=*/64);
	// uint64_t descB = make_smem_desc(sB, /*ld_major=*/32, /*ld_minor=*/8);

	// 16
	//uint64_t descA = make_smem_desc(sA, 64, 32);
	//uint64_t descB = make_smem_desc(sB, 32,  8);

	// 16
	//uint64_t descA = make_smem_desc(sA, 32,  64);
	//uint64_t descB = make_smem_desc(sB, 8,  32);

	// 0
	//uint64_t descA = make_smem_desc(sA, 32,  1024);
	//uint64_t descB = make_smem_desc(sB, 32,  1024);

	// 0
	// uint64_t descA = make_smem_desc(sA, 32,  2048);
	// uint64_t descB = make_smem_desc(sB, 32,  128);


	//-----------//
	// try2 :
	//-----------//
	// 2
	//uint64_t descA = make_smem_desc(sA, 32, 64);
	//uint64_t descB = make_smem_desc(sB, 32,  8);

	//uint64_t descA = make_smem_desc(sA, 32, 32);
	//uint64_t descB = make_smem_desc(sB, 32, 32);


	//-----------//
	// try3 :
	//-----------//
	// 1
	//uint64_t descA = make_smem_desc(sA, 32, 64);
	//uint64_t descB = make_smem_desc(sB, 32,  8);

	// 2
	// uint64_t descA = make_smem_desc(sA, 64, 32);
	// uint64_t descB = make_smem_desc(sB, 8,  32);

	//-----------//
	// others: 
	//-----------//

	// uint64_t descA = make_smem_desc(sA, /*ld_major=*/32, /*ld_minor=*/64);
	// uint64_t descB = make_smem_desc(sB, /*ld_major=*/32, /*ld_minor=*/8);

	// uint64_t descA = make_smem_desc(sA, /*ld_major=*/64, /*ld_minor=*/32);
	// uint64_t descB = make_smem_desc(sB, /*ld_major=*/32, /*ld_minor=*/8);

	//uint64_t descA = make_smem_desc(sA, /*ld_major=*/64, /*ld_minor=*/32);
	//uint64_t descB = make_smem_desc(sB, /*ld_major=*/8, /*ld_minor=*/32);

	// Our accumulators: 2 x 32-bit registers => 4 total fp16 values
	// C is 64x8 , each warp read 16x8 of inputC, there are 32 threads per warp, so each fiber hold 4 input values.
	// ideally, we would like fiber0 holds the 1st 2 inputs, the offset by 8x8 elements, hold the 2nd 2 inputs.

	// here, fiber reads the 1st 2xfp16, and next 2xfp16
	// noted that, we only care about the c0 input 
	uint32_t c0 = C[2*tid + 0];
	uint32_t c1 = C[2*tid + 1];

	// Start the WGMMA group
	wgmma_fence();

	// Choose WGMMA implementation based on format
	if constexpr (FORMAT == FP8Format::E5M2) {
		wgmma_m64n8k32_f16_e5m2_e5m2<1,1,1>(descA, descB, c0, c1);
	} else {
		wgmma_m64n8k32_f16_e4m3_e4m3<1,1,1>(descA, descB, c0, c1);
	}

	// Commit, then wait for group 0
	wgmma_commit_group();
	wgmma_wait_group<0>();

	// Now c0,c1 each hold two FP16 accumulators. In a real code,
	// you'd map lane IDs carefully to store each portion of the 64x8 tile.
	// For demonstration, we simply store c0,c1 from each thread to global mem.

	// matrix C is 64x8
	int store_idx = tid; // 0..127
	if(tid == 0) {
		C[tid] = c0;
	}
	// C[2 * store_idx + 0] = c0;
	//C[2 * store_idx + 1] = c1;

	// print the lower half of c0
	uint16_t c0_lo = static_cast<uint16_t>(c0 & 0xFFFF);
        uint16_t c0_hi = static_cast<uint16_t>((c0 >> 16) & 0xFFFF);
#if DEBUG
	if(tid == 0) {
		printf("kernel: tid=%d c0_lo=0x%4X \n", tid, c0_lo);
		// printf("[tid=%d] c0_lo=0x%4X c0_hi=0x%4X  \n", tid, c0_lo, c0_hi);
	}
#endif
}

//----------------------------------------------------------------------------//
// Main 
//----------------------------------------------------------------------------//
int main(int argc, char **argv)
{
	//------------------------------------------------------------------------//
	// Read commandlines
	//------------------------------------------------------------------------//
	if (argc != 4)
	{
		std::cerr << "\nUsage: " << argv[0] << " <filename> <format> <device_id>" << std::endl;
		std::cerr << "  format: e5m2 or e4m3" << std::endl;
		std::cerr << "  device_id: GPU device ID (0 to N-1)" << std::endl;
		return 1;
	}

	// Parse the format argument
	std::string format_str = argv[2];
	bool use_e5m2;
	if (format_str == "e5m2") {
		use_e5m2 = true;
	} else if (format_str == "e4m3") {
		use_e5m2 = false;
	} else {
		std::cerr << "Error: format must be either 'e5m2' or 'e4m3'" << std::endl;
		return 1;
	}

	// Parse and set the device ID
	int device_id = std::stoi(argv[3]);
	int device_count;
	cudaCheck(cudaGetDeviceCount(&device_count));
	
	if (device_id < 0 || device_id >= device_count) {
		std::cerr << "Error: device_id must be between 0 and " << (device_count - 1) << std::endl;
		return 1;
	}

	cudaCheck(cudaSetDevice(device_id));
	
	// Get and print device properties
	cudaDeviceProp prop;
	cudaCheck(cudaGetDeviceProperties(&prop, device_id));
	logMessage("Using GPU device %d: %s", device_id, prop.name);

	//------------------------------------------------------------------------//
	// Read all test cases
	//------------------------------------------------------------------------//
	// 65 input values per row:   c +  32 of a/b for K32 case
	// 33 input values per row:   c +  16 of a/b for K16 case
	// 17 input values per row:   c +   8 of a/b for  K8 case
	std::vector<uint16_t> allTests_c;               // input c in f16
	std::vector<std::vector<uint8_t>> allTests_ab;  // input a/b in fp8

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

		// Check if this is a K8/K16/K32 case
		if (numbers_ab.size() == 32) {  // K16 case
			// logMessage("K16 case detected - padded with zeros");
			numbers_ab.resize(64, 0);  // Add 32 zeros
		}
		else if (numbers_ab.size() == 16) {  // K8 case
			// logMessage("K8 case detected - padded with zeros");
			numbers_ab.resize(64, 0);  // Add zeros
		}
		else if (numbers_ab.size() != 64) {
			std::cerr << "Error: Invalid input line length. Expected 16 or 32 or 64 values for A/B, got " 
					  << numbers_ab.size() << std::endl;
			return 1;
		}

		allTests_ab.push_back(numbers_ab); // store current line for a and b
	}

	std::cout << std::endl;

	file.close();

	//------------------------------------------------------------------------//
    // Check first line : c + 32x{a(i), b(i)}
    //------------------------------------------------------------------------//
#if 1 
    printf("\nCheck first line of input file:\n");
    printf("%04X ", allTests_c[0]);
    for (int i = 0; i < 64; i++)
    {
        printf("%02X ", allTests_ab[0][i]);
    }
     printf("\n");
#endif

    //------------------------------------------------------------------------//
    // Run all test cases
    //------------------------------------------------------------------------//
    std::cout << "Run Tensor Core Tests with " << format_str << " format\n" << std::endl;

    // Change from int to size_t for totalNum
    size_t totalNum = allTests_ab.size();

    // results in fp16 
    std::vector<std::vector<uint16_t>> allTests_results(totalNum);

    for (size_t i = 0; i < totalNum; i++)  // Change loop variable type to match
    {

	if((i%500) == 0)  logMessage("case : %zu (%zu) \n ... \n", i, totalNum);  // Change format specifier from %d to %zu for size_t

		// each test inputs
		uint16_t current_test_c = allTests_c[i];
		std::vector<uint8_t> current_test_ab = allTests_ab[i];

		// output in fp16
		std::vector<uint16_t> current_result;

		//--------------------------------------------------------------------//
		// run tensor core test
		//--------------------------------------------------------------------//
		if (use_e5m2) {  // Add a command line argument or configuration to set this
			runTest<K32, FP8Format::E5M2>(current_test_ab, current_test_c, current_result);
		} else {
			runTest<K32, FP8Format::E4M3>(current_test_ab, current_test_c, current_result);
		}

		allTests_results[i] = current_result;
	}

    //------------------------------------------------------------------------//
    // Save to txt file 
    //------------------------------------------------------------------------//
    std::string outFileName = "gpu_output.txt";
    std::ofstream outFile(outFileName);
    if (!outFile)
    {
        std::cerr << "Error opening file to write." << std::endl;
        return 1;
    }

    const int colNum = 1;
    for (size_t i = 0; i < totalNum; i++)
    {
        for (int j = 0; j < colNum; j++)
        {
            outFile << std::setfill('0') << std::setw(4) << std::hex << allTests_results[i][j];
            // if (j < colNum) outFile << " "; // separate with space
        }
        outFile << "\n"; // EOR
    }

    outFile.close();

    std::cout << "\nResults are saved! Check " << outFileName << ".\n";

    return 0;
};

//----------------------------------------------------------------------------//
// host code to prepare test
//----------------------------------------------------------------------------//
template <int TILE_K, FP8Format FORMAT>
void runTest(std::vector<uint8_t> current_test_ab,
			 uint16_t current_test_c,
			 std::vector<uint16_t> &current_result)
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
	size_t sizeD = M * N;   // noted: for fp16 accumulation, each 32 reg will store two fp16 results

	hA = (uint8_t *)malloc(sizeof(uint8_t) * sizeA);
	hB = (uint8_t *)malloc(sizeof(uint8_t) * sizeB);
	hD = (uint32_t *)malloc(sizeof(uint32_t) * sizeD);
	hresult = (uint32_t *)malloc(sizeof(uint32_t) * sizeD);

	// init to 0
	memset(hA, 0, sizeof(uint8_t) * sizeA);  // MxK
	memset(hB, 0, sizeof(uint8_t) * sizeB); // NxK
	memset(hD, 0, sizeof(uint32_t) * sizeD); // MxN
	memset(hresult, 0, sizeof(uint32_t) * sizeD);

	//------------------------------------------------------------------------//
	// read/set up data on cpu
	//------------------------------------------------------------------------//
#if DEBUG
	std::cout << "Read inputs a/b " << std::endl;
#endif
	for (int i = 0; i < TILE_K; i++)
	{
        // MxK:   only first 32 elements are intiliazed, the rest are 0
		hA[i] = current_test_ab[i * 2];		//  read a

        // NxK
		uint8_t val_b = current_test_ab[i * 2 + 1]; //  read b
        hB[i] = val_b;
	}


#if DEBUG
	std::cout << "Read input C" << std::endl;
#endif
	// pack fp16 into a 32 bit register
	uint32_t packed = 0;
	packed |= (uint32_t)current_test_c & 0xFFFF;    // put low half 
	// packed |= ((uint32_t)half_hi & 0xFFFF) << 16;    // put high half 
	hD[0] = packed; 

#if DEBUG
    printf("Pack input C (fp16) into 32bit register : %08X \n\n", hD[0]);
#endif



	//------------------------------------------------------------------------//
	// gpu buffer
	//------------------------------------------------------------------------//
	uint8_t *dA = nullptr;
	uint8_t *dB = nullptr;
	uint32_t *dD = nullptr;

	cudaMalloc((void **)&dA, sizeof(uint8_t) * sizeA);
	cudaMalloc((void **)&dB, sizeof(uint8_t) * sizeB);
	cudaMalloc((void **)&dD, sizeof(uint32_t) * sizeD);

	// h2d : copy input ops to gpu
	cudaMemcpy(dA, hA, sizeof(uint8_t) * sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(uint8_t) * sizeB, cudaMemcpyHostToDevice);
	cudaMemcpy(dD, hD, sizeof(uint32_t) * sizeD, cudaMemcpyHostToDevice);

	// Launch a single block with 128 threads => "1 warpgroup" (4 warps, 32 threads per warp)
    matmul_fp8_64x8x32_kernel<FORMAT><<<1, 128>>>(dA, dB, dD);

	// d2h : copy results back to host
	cudaMemcpy(hresult, dD, sizeof(uint32_t) * sizeD, cudaMemcpyDeviceToHost);

	uint32_t c0 = hresult[0];
	uint16_t c0_lo = static_cast<uint16_t>(c0 & 0xFFFF);   // read the lower half (1st 16 bits)
	// uint16_t c0_hi = static_cast<uint16_t>((c0 >> 16) & 0xFFFF);

// check value
#if DEBUG
	printf("%08X\n", hresult[0]);
	printf("c0_lo=0x%04X \n", c0_lo);
	// printf("[tid=0] c0_lo=0x%04X c0_hi=0x%04X \n", c0_lo, c0_hi);
#endif

	printf("\n\n");

	current_result.push_back(c0_lo);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dD);

	free(hA);
	free(hB);
	free(hD);
	free(hresult);

	cudaDeviceSynchronize();
}
