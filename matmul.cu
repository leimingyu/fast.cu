#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


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
#include <unistd.h>

#include <iomanip>  // setfill, setw
#include <cstdint>  // int8_t/uint8_t

#define DEBUG 1
#define K32 32 


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(1);
  }
}

// typedef __nv_bfloat16 bf16;
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

template <int TILE_K>
void runTest(std::vector<uint8_t> current_test_ab,
             uint32_t current_test_c,
             std::vector<uint32_t> &current_result);

//----------------------------------------------------------------------------//
// wgmma related apis 
//----------------------------------------------------------------------------//

// wgmma example in bf16
// #include "examples/matmul/matmul_2.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

// __device__ uint64_t make_smem_desc(bf16* ptr) {
//     uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
//     uint64_t desc = 0x0000000000000000;
//     desc |= matrix_descriptor_encode(addr);
//     desc |= matrix_descriptor_encode((uint64_t)16) << 16;
//     desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
//     desc |= 1llu << 62; // 128B swizzle
//     return desc;
//   }

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// template <int BlockMajorSize, int BlockMinorSize>
// void create_tensor_map(CUtensorMap *tma_map, bf16* gmem_ptr, int blocks_height, int blocks_width) {
//     void* gmem_address = (void*)gmem_ptr;
//     uint64_t gmem_prob_shape[5] = {(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1};
//     uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize*blocks_width, 0, 0, 0};
//     uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};
//     uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

//     CUresult result = cuTensorMapEncodeTiled(
//         tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
//         gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
//         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

//     assert(result == CUDA_SUCCESS);
// }



// template <int BlockMajorSize, int BlockMinorSize>
// __host__ static inline CUtensorMap* allocate_and_create_tensor_map(bf16* src, int blocks_height, int blocks_width) {
//     CUtensorMap *tma_map_d;
//     cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
//     CUtensorMap tma_map_host;
//     create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
//     cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
//     return tma_map_d;
// }

//----------------------------------------------------------------------------//
// gpu tests and kernels
//----------------------------------------------------------------------------//



/*
__global__ void kernel_wgmma_FP8(float *buf_fp32, half *buf_fp16,
                               uint8_t *test_ops, uint32_t *test_init_c, uint32_t *result_gpu)
{
  //------------------------------------------------------------------------//
  // registers for mma
  //------------------------------------------------------------------------//
  float D[4] = {0.f, 0.f, 0.f, 0.f};
  float C[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t A[4]; // A0, A1, A2, A3
  uint32_t B[2];

  //------------------------------------------------------------------------//
  // Init A/B to zeros
  //------------------------------------------------------------------------//
  A[0] = 0;
  A[1] = 0;
  A[2] = 0;
  A[3] = 0;

  B[0] = 0;
  B[1] = 0;

  //------------------------------------------------------------------------//
  // first mma call :   C = 0
  //------------------------------------------------------------------------//

  //------------------------------------------------------------------------//
  // Init inputs
  //------------------------------------------------------------------------//
  uint8_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31;
  uint8_t b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29, b30, b31;

  // set a to 0
  a0 = 0;
  a1 = 0;
  a2 = 0;
  a3 = 0;
  a4 = 0;
  a5 = 0;
  a6 = 0;
  a7 = 0;
  a8 = 0;
  a9 = 0;
  a10 = 0;
  a11 = 0;
  a12 = 0;
  a13 = 0;
  a14 = 0;
  a15 = 0;
  a16 = 0;
  a17 = 0;
  a18 = 0;
  a19 = 0;
  a20 = 0;
  a21 = 0;
  a22 = 0;
  a23 = 0;
  a24 = 0;
  a25 = 0;
  a26 = 0;
  a27 = 0;
  a28 = 0;
  a29 = 0;
  a30 = 0;
  a31 = 0;

  // set b to 0
  b0 = 0;
  b1 = 0;
  b2 = 0;
  b3 = 0;
  b4 = 0;
  b5 = 0;
  b6 = 0;
  b7 = 0;
  b8 = 0;
  b9 = 0;
  b10 = 0;
  b11 = 0;
  b12 = 0;
  b13 = 0;
  b14 = 0;
  b15 = 0;
  b16 = 0;
  b17 = 0;
  b18 = 0;
  b19 = 0;
  b20 = 0;
  b21 = 0;
  b22 = 0;
  b23 = 0;
  b24 = 0;
  b25 = 0;
  b26 = 0;
  b27 = 0;
  b28 = 0;
  b29 = 0;
  b30 = 0;
  b31 = 0;

  //--- a0 x b0 ---//
  a0 = (a0 | test_ops[0]);
  b0 = (b0 | test_ops[1]);
  //--- a1 x b1 ---//
  a1 = (a1 | test_ops[2]);
  b1 = (b1 | test_ops[3]);
  //--- a2 x b2 ---//
  a2 = (a2 | test_ops[4]);
  b2 = (b2 | test_ops[5]);
  //--- a3 x b3 ---//
  a3 = (a3 | test_ops[6]);
  b3 = (b3 | test_ops[7]);

  //--- a4 x b4 ---//
  a4 = (a4 | test_ops[8]);
  b4 = (b4 | test_ops[9]);
  //--- a5 x b5 ---//
  a5 = (a5 | test_ops[10]);
  b5 = (b5 | test_ops[11]);
  //--- a6 x b6 ---//
  a6 = (a6 | test_ops[12]);
  b6 = (b6 | test_ops[13]);
  //--- a7 x b7 ---//
  a7 = (a7 | test_ops[14]);
  b7 = (b7 | test_ops[15]);

  //--- a8 x b8 ---//
  a8 = (a8 | test_ops[16]);
  b8 = (b8 | test_ops[17]);
  //--- a9 x b9 ---//
  a9 = (a9 | test_ops[18]);
  b9 = (b9 | test_ops[19]);
  //--- a10 x b10 ---//
  a10 = (a10 | test_ops[20]);
  b10 = (b10 | test_ops[21]);
  //--- a11 x b11 ---//
  a11 = (a11 | test_ops[22]);
  b11 = (b11 | test_ops[23]);

  //--- a12 x b12 ---//
  a12 = (a12 | test_ops[24]);
  b12 = (b12 | test_ops[25]);
  //--- a13 x b13 ---//
  a13 = (a13 | test_ops[26]);
  b13 = (b13 | test_ops[27]);
  //--- a14 x b14 ---//
  a14 = (a14 | test_ops[28]);
  b14 = (b14 | test_ops[29]);
  //--- a15 x b15 ---//
  a15 = (a15 | test_ops[30]);
  b15 = (b15 | test_ops[31]);

  //--- a16 x b16 ---//
  a16 = (a16 | test_ops[32]);
  b16 = (b16 | test_ops[33]);
  //--- a17 x b17 ---//
  a17 = (a17 | test_ops[34]);
  b17 = (b17 | test_ops[35]);
  //--- a18 x b18 ---//
  a18 = (a18 | test_ops[36]);
  b18 = (b18 | test_ops[37]);
  //--- a19 x b19 ---//
  a19 = (a19 | test_ops[38]);
  b19 = (b19 | test_ops[39]);

  //--- a20 x b20 ---//
  a20 = (a20 | test_ops[40]);
  b20 = (b20 | test_ops[41]);
  //--- a21 x b21 ---//
  a21 = (a21 | test_ops[42]);
  b21 = (b21 | test_ops[43]);
  //--- a22 x b22 ---//
  a22 = (a22 | test_ops[44]);
  b22 = (b22 | test_ops[45]);
  //--- a23 x b23 ---//
  a23 = (a23 | test_ops[46]);
  b23 = (b23 | test_ops[47]);

  //--- a24 x b24 ---//
  a24 = (a24 | test_ops[48]);
  b24 = (b24 | test_ops[49]);
  //--- a25 x b25 ---//
  a25 = (a25 | test_ops[50]);
  b25 = (b25 | test_ops[51]);
  //--- a26 x b26 ---//
  a26 = (a26 | test_ops[52]);
  b26 = (b26 | test_ops[53]);
  //--- a27 x b27 ---//
  a27 = (a27 | test_ops[54]);
  b27 = (b27 | test_ops[55]);

  //--- a28 x b28 ---//
  a28 = (a28 | test_ops[56]);
  b28 = (b28 | test_ops[57]);
  //--- a29 x b29 ---//
  a29 = (a29 | test_ops[58]);
  b29 = (b29 | test_ops[59]);
  //--- a30 x b30 ---//
  a30 = (a30 | test_ops[60]);
  b30 = (b30 | test_ops[61]);
  //--- a31 x b31 ---//
  a31 = (a31 | test_ops[62]);
  b31 = (b31 | test_ops[63]);

  // Pass the input ops to tc inputs
  uint8_t a0_t0[4] = {a0, a1, a2, a3};     // A[0] for T0
  uint8_t a0_t1[4] = {a4, a5, a6, a7};     // A[0] for T1
  uint8_t a0_t2[4] = {a8, a9, a10, a11};   // A[0] for T2
  uint8_t a0_t3[4] = {a12, a13, a14, a15}; // A[0] for T3

  uint8_t a2_t0[4] = {a16, a17, a18, a19}; // A[2] for T0
  uint8_t a2_t1[4] = {a20, a21, a22, a23}; // A[2] for T1
  uint8_t a2_t2[4] = {a24, a25, a26, a27}; // A[2] for T2
  uint8_t a2_t3[4] = {a28, a29, a30, a31}; // A[2] for T3

  uint8_t b0_t0[4] = {b0, b1, b2, b3};     // B[0] for T0
  uint8_t b0_t1[4] = {b4, b5, b6, b7};     // B[0] for T1
  uint8_t b0_t2[4] = {b8, b9, b10, b11};   // B[0] for T2
  uint8_t b0_t3[4] = {b12, b13, b14, b15}; // B[0] for T3

  uint8_t b1_t0[4] = {b16, b17, b18, b19}; // B[1] for T0
  uint8_t b1_t1[4] = {b20, b21, b22, b23}; // B[1] for T1
  uint8_t b1_t2[4] = {b24, b25, b26, b27}; // B[1] for T2
  uint8_t b1_t3[4] = {b28, b29, b30, b31}; // B[1] for T3

  if (threadIdx.x == 0)
  {
    memcpy(&A[0], a0_t0, 4);
    memcpy(&A[2], a2_t0, 4);

    memcpy(&B[0], b0_t0, 4);
    memcpy(&B[1], b1_t0, 4);
  }

  if (threadIdx.x == 1)
  {
    memcpy(&A[0], a0_t1, 4);
    memcpy(&A[2], a2_t1, 4);

    memcpy(&B[0], b0_t1, 4);
    memcpy(&B[1], b1_t1, 4);
  }

  if (threadIdx.x == 2)
  {
    memcpy(&A[0], a0_t2, 4);
    memcpy(&A[2], a2_t2, 4);

    memcpy(&B[0], b0_t2, 4);
    memcpy(&B[1], b1_t2, 4);
  }

  if (threadIdx.x == 3)
  {
    memcpy(&A[0], a0_t3, 4);
    memcpy(&A[2], a2_t3, 4);

    memcpy(&B[0], b0_t3, 4);
    memcpy(&B[1], b1_t3, 4);
  }

#if USE_E4M3
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

#if USE_E5M2
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

  // Read result directly from D[0]
  fp32_ tc_result;
  tc_result.f = D[0];

  // check result using T0
#if DEBUG
  if (threadIdx.x == 0)
  {
    printf("Tensor Core result : %.20f, hex=%08X\n", tc_result.f, tc_result.i);
  }
#endif

  __syncthreads(); // ensure the 1st mma is done

  //------------------------------------------------------------------------//
  // second mma call :   add C
  //------------------------------------------------------------------------//

  // reset
  D[0] = 0.f;
  D[1] = 0.f;
  D[2] = 0.f;
  D[3] = 0.f;
  C[0] = 0.f;
  C[1] = 0.f;
  C[2] = 0.f;
  C[3] = 0.f;
  A[0] = 0;
  A[1] = 0;
  A[2] = 0;
  A[3] = 0;
  B[0] = 0;
  B[1] = 0;

  // Set Init C
  fp32_ c_fp32_hex;
  c_fp32_hex.i = test_init_c[0];
  C[0] = c_fp32_hex.f;

  // set a to 0
  a0 = 0;
  a1 = 0;
  a2 = 0;
  a3 = 0;
  a4 = 0;
  a5 = 0;
  a6 = 0;
  a7 = 0;
  a8 = 0;
  a9 = 0;
  a10 = 0;
  a11 = 0;
  a12 = 0;
  a13 = 0;
  a14 = 0;
  a15 = 0;

  a16 = 0;
  a17 = 0;
  a18 = 0;
  a19 = 0;
  a20 = 0;
  a21 = 0;
  a22 = 0;
  a23 = 0;
  a24 = 0;
  a25 = 0;
  a26 = 0;
  a27 = 0;
  a28 = 0;
  a29 = 0;
  a30 = 0;
  a31 = 0;

  // set b to 0
  b0 = 0;
  b1 = 0;
  b2 = 0;
  b3 = 0;
  b4 = 0;
  b5 = 0;
  b6 = 0;
  b7 = 0;
  b8 = 0;
  b9 = 0;
  b10 = 0;
  b11 = 0;
  b12 = 0;
  b13 = 0;
  b14 = 0;
  b15 = 0;

  b16 = 0;
  b17 = 0;
  b18 = 0;
  b19 = 0;
  b20 = 0;
  b21 = 0;
  b22 = 0;
  b23 = 0;
  b24 = 0;
  b25 = 0;
  b26 = 0;
  b27 = 0;
  b28 = 0;
  b29 = 0;
  b30 = 0;
  b31 = 0;

  //--- a0 x b0 ---//
  a0 = (a0 | test_ops[0]);
  b0 = (b0 | test_ops[1]);
  //--- a1 x b1 ---//
  a1 = (a1 | test_ops[2]);
  b1 = (b1 | test_ops[3]);
  //--- a2 x b2 ---//
  a2 = (a2 | test_ops[4]);
  b2 = (b2 | test_ops[5]);
  //--- a3 x b3 ---//
  a3 = (a3 | test_ops[6]);
  b3 = (b3 | test_ops[7]);

  //--- a4 x b4 ---//
  a4 = (a4 | test_ops[8]);
  b4 = (b4 | test_ops[9]);
  //--- a5 x b5 ---//
  a5 = (a5 | test_ops[10]);
  b5 = (b5 | test_ops[11]);
  //--- a6 x b6 ---//
  a6 = (a6 | test_ops[12]);
  b6 = (b6 | test_ops[13]);
  //--- a7 x b7 ---//
  a7 = (a7 | test_ops[14]);
  b7 = (b7 | test_ops[15]);

  //--- a8 x b8 ---//
  a8 = (a8 | test_ops[16]);
  b8 = (b8 | test_ops[17]);
  //--- a9 x b9 ---//
  a9 = (a9 | test_ops[18]);
  b9 = (b9 | test_ops[19]);
  //--- a10 x b10 ---//
  a10 = (a10 | test_ops[20]);
  b10 = (b10 | test_ops[21]);
  //--- a11 x b11 ---//
  a11 = (a11 | test_ops[22]);
  b11 = (b11 | test_ops[23]);

  //--- a12 x b12 ---//
  a12 = (a12 | test_ops[24]);
  b12 = (b12 | test_ops[25]);
  //--- a13 x b13 ---//
  a13 = (a13 | test_ops[26]);
  b13 = (b13 | test_ops[27]);
  //--- a14 x b14 ---//
  a14 = (a14 | test_ops[28]);
  b14 = (b14 | test_ops[29]);
  //--- a15 x b15 ---//
  a15 = (a15 | test_ops[30]);
  b15 = (b15 | test_ops[31]);

  //--- a16 x b16 ---//
  a16 = (a16 | test_ops[32]);
  b16 = (b16 | test_ops[33]);
  //--- a17 x b17 ---//
  a17 = (a17 | test_ops[34]);
  b17 = (b17 | test_ops[35]);
  //--- a18 x b18 ---//
  a18 = (a18 | test_ops[36]);
  b18 = (b18 | test_ops[37]);
  //--- a19 x b19 ---//
  a19 = (a19 | test_ops[38]);
  b19 = (b19 | test_ops[39]);

  //--- a20 x b20 ---//
  a20 = (a20 | test_ops[40]);
  b20 = (b20 | test_ops[41]);
  //--- a21 x b21 ---//
  a21 = (a21 | test_ops[42]);
  b21 = (b21 | test_ops[43]);
  //--- a22 x b22 ---//
  a22 = (a22 | test_ops[44]);
  b22 = (b22 | test_ops[45]);
  //--- a23 x b23 ---//
  a23 = (a23 | test_ops[46]);
  b23 = (b23 | test_ops[47]);

  //--- a24 x b24 ---//
  a24 = (a24 | test_ops[48]);
  b24 = (b24 | test_ops[49]);
  //--- a25 x b25 ---//
  a25 = (a25 | test_ops[50]);
  b25 = (b25 | test_ops[51]);
  //--- a26 x b26 ---//
  a26 = (a26 | test_ops[52]);
  b26 = (b26 | test_ops[53]);
  //--- a27 x b27 ---//
  a27 = (a27 | test_ops[54]);
  b27 = (b27 | test_ops[55]);

  //--- a28 x b28 ---//
  a28 = (a28 | test_ops[56]);
  b28 = (b28 | test_ops[57]);
  //--- a29 x b29 ---//
  a29 = (a29 | test_ops[58]);
  b29 = (b29 | test_ops[59]);
  //--- a30 x b30 ---//
  a30 = (a30 | test_ops[60]);
  b30 = (b30 | test_ops[61]);
  //--- a31 x b31 ---//
  a31 = (a31 | test_ops[62]);
  b31 = (b31 | test_ops[63]);

  // Pass the input ops to tc inputs
  uint8_t a0_t0_[4] = {a0, a1, a2, a3};     // A[0] for T0
  uint8_t a0_t1_[4] = {a4, a5, a6, a7};     // A[0] for T1
  uint8_t a0_t2_[4] = {a8, a9, a10, a11};   // A[0] for T2
  uint8_t a0_t3_[4] = {a12, a13, a14, a15}; // A[0] for T3

  uint8_t a2_t0_[4] = {a16, a17, a18, a19}; // A[2] for T0
  uint8_t a2_t1_[4] = {a20, a21, a22, a23}; // A[2] for T1
  uint8_t a2_t2_[4] = {a24, a25, a26, a27}; // A[2] for T2
  uint8_t a2_t3_[4] = {a28, a29, a30, a31}; // A[2] for T3

  uint8_t b0_t0_[4] = {b0, b1, b2, b3};     // B[0] for T0
  uint8_t b0_t1_[4] = {b4, b5, b6, b7};     // B[0] for T1
  uint8_t b0_t2_[4] = {b8, b9, b10, b11};   // B[0] for T2
  uint8_t b0_t3_[4] = {b12, b13, b14, b15}; // B[0] for T3

  uint8_t b1_t0_[4] = {b16, b17, b18, b19}; // B[1] for T0
  uint8_t b1_t1_[4] = {b20, b21, b22, b23}; // B[1] for T1
  uint8_t b1_t2_[4] = {b24, b25, b26, b27}; // B[1] for T2
  uint8_t b1_t3_[4] = {b28, b29, b30, b31}; // B[1] for T3

  if (threadIdx.x == 0)
  {
    memcpy(&A[0], a0_t0_, 4);
    memcpy(&A[2], a2_t0_, 4);

    memcpy(&B[0], b0_t0_, 4);
    memcpy(&B[1], b1_t0_, 4);
  }

  if (threadIdx.x == 1)
  {
    memcpy(&A[0], a0_t1_, 4);
    memcpy(&A[2], a2_t1_, 4);

    memcpy(&B[0], b0_t1_, 4);
    memcpy(&B[1], b1_t1_, 4);
  }

  if (threadIdx.x == 2)
  {
    memcpy(&A[0], a0_t2_, 4);
    memcpy(&A[2], a2_t2_, 4);

    memcpy(&B[0], b0_t2_, 4);
    memcpy(&B[1], b1_t2_, 4);
  }

  if (threadIdx.x == 3)
  {
    memcpy(&A[0], a0_t3_, 4);
    memcpy(&A[2], a2_t3_, 4);

    memcpy(&B[0], b0_t3_, 4);
    memcpy(&B[1], b1_t3_, 4);
  }

#if USE_E4M3
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

#if USE_E5M2
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
#endif

  // Read result directly from D[0]
  fp32_ tc_result1;
  tc_result1.f = D[0];

  // check result using T0
#if DEBUG
  if (threadIdx.x == 0)
  {
    printf("Tensor Core result : %.20f, hex=%08X\n", tc_result1.f, tc_result1.i);
  }
#endif

  //------------------------------------------------------------------------//
  // save tc results
  //------------------------------------------------------------------------//
  if (threadIdx.x == 0)
  {
    result_gpu[0] = tc_result.i;
    result_gpu[1] = tc_result1.i;
  }

  //---------------------//
  // avoid optimze code away
  //---------------------//
  C[0] += D[0];
  C[1] += D[1];
  C[1] += C[0];
  C[0] += C[1];

  // copy c0,c1
  memcpy(&buf_fp32[threadIdx.x * 4], C, 16); // 4 of 32 bits = 16 bytes = 4 fp32 values
}
*/

// __global__ void warmupKernel() {
//   __shared__ int s[100];
//   s[0] += s[1];
// }



// input file 
// 65 input per row:   c +  32 of a/b
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
  std::vector<uint32_t> allTests_c;              // input c
  std::vector<std::vector<uint8_t>> allTests_ab; // input a/b

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
        std::vector<uint8_t> numbers_ab;  // current line for a/b
        std::string hexStr;

        // read c first
        iss >> hexStr;
        uint32_t num_c = static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16));
        allTests_c.push_back(num_c);  // store current line for C

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
    printf("%08X ", allTests_c[0]);
    for (int i = 0; i < 64; i++)
    {
        printf("%02X ", allTests_ab[0][i]);
    }
    printf("\n\n");
#endif

    //------------------------------------------------------------------------//
    // Run all test cases
    //------------------------------------------------------------------------//
    std::cout << "\nRun Tensor Core Tests" << std::endl;

    // prepare results
    int totalNum = static_cast<int>(allTests_ab.size());
    std::vector<std::vector<uint32_t>> allTests_results(totalNum);

    for (int i = 0; i < totalNum; i++)
    {
      // each test inputs
      uint32_t current_test_c = allTests_c[i];
      std::vector<uint8_t> current_test_ab = allTests_ab[i];

      // output
      std::vector<uint32_t> current_result;

      //--------------------------------------------------------------------//
      // run tensor core test
      //--------------------------------------------------------------------//
      runTest<K32>(current_test_ab, current_test_c, current_result);

#if DEBUG
      // printf("%08X %08X\n", current_result[0], current_result[1]);
#endif
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
			 uint32_t current_test_c,
			 std::vector<uint32_t> &current_result)
{
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

	uint32_t *hC = nullptr;
	uint32_t *hD = nullptr;

	size_t sizeA = M * K;
	size_t sizeB = N * K;
	size_t sizeCD = M * N;

	hA = (uint8_t *)malloc(sizeof(uint8_t) * sizeA);
	hB = (uint8_t *)malloc(sizeof(uint8_t) * sizeB);

	hC = (uint32_t *)malloc(sizeof(uint32_t) * sizeCD);
	hD = (uint32_t *)malloc(sizeof(uint32_t) * sizeCD);

	// init to 0
	memset(hA, 0, sizeof(uint8_t) * sizeA);
	memset(hB, 0, sizeof(uint8_t) * sizeB);

	memset(hC, 0, sizeof(uint32_t) * sizeCD);
	memset(hD, 0, sizeof(uint32_t) * sizeCD);

	//------------------------------------------------------------------------//
	// read/set up data on cpu
	//------------------------------------------------------------------------//
	std::cout << "Read inputs a/b " << std::endl;
	for (int i = 0; i < TILE_K; i++)
	{
		hA[i] = current_test_ab[i * 2];		//  read a
		hB[i] = current_test_ab[i * 2 + 1]; //  read b
	}

	std::cout << "Read input C" << std::endl;
	hC[0] = current_test_c;

	// // cpu buffer: two results in fp32
	// // 1st : dotproduct of (a, b)
	// // 2nd : dotproduct of (a, b) + C
	// uint32_t *result_cpu = 0;
	// result_cpu = (uint32_t *)malloc(sizeof(uint32_t) * 2);

	//------------------------------------------------------------------------//
	// gpu buffer
	//------------------------------------------------------------------------//
	uint8_t *dA = nullptr;
	uint8_t *dB = nullptr;
	uint32_t *dC = nullptr;
	uint32_t *dD = nullptr;

	cudaMalloc((void **)&dA, sizeof(uint8_t) * sizeA);
	cudaMalloc((void **)&dB, sizeof(uint8_t) * sizeB);

	cudaMalloc((void **)&dC, sizeof(uint32_t) * sizeCD);
	cudaMalloc((void **)&dD, sizeof(uint32_t) * sizeCD);

	// float *buf_fp32 = 0;
	// half *buf_fp16 = 0;
	// cudaMalloc((void **)&buf_fp32, sizeof(float) * 1024);
	// cudaMalloc((void **)&buf_fp16, sizeof(half) * 1024);


	// // output buffer
	// cudaMalloc((void **)&result_gpu, sizeof(uint32_t) * 2);

	// h2d : copy input ops to gpu
	cudaMemcpy(dA, hA, sizeof(uint8_t) * sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(uint8_t) * sizeB, cudaMemcpyHostToDevice);

	cudaMemcpy(dC, hC, sizeof(uint32_t) * sizeCD, cudaMemcpyHostToDevice);
	cudaMemcpy(dD, hD, sizeof(uint32_t) * sizeCD, cudaMemcpyHostToDevice);

	constexpr int BM = 64;
	constexpr int BN = 8;
	constexpr int BK = 32;
	constexpr int NUM_THREADS = 128;

	CUtensorMap *d_tma_map_A = 0;
	CUtensorMap *d_tma_map_B = 0;

	//   d_tma_map_A = allocate_and_create_tensor_map<BM, BK>(dA, m / BM, k / BK);
	//   d_tma_map_B = allocate_and_create_tensor_map<BN, BK>(dB, m / BN, k / BK);

	//------------------------------------------------------------------------//
	// 1 warpgroup = 128 threads
	//------------------------------------------------------------------------//
	//   kernel_wgmma_FP8<<<1, 128>>>(buf_fp32, buf_fp16, test_inputs_ab, test_inputs_c, result_gpu);

	/*
	  // d2h : copy results back to host
	  cudaMemcpy(result_cpu, result_gpu, sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost);

	  // check value
	#if DEBUG
	  printf("%08X %08X\n", result_cpu[0], result_cpu[1]);
	#endif

	  current_result.push_back(result_cpu[0]);
	  current_result.push_back(result_cpu[1]);

	  if (buf_fp32)
	  {
		cudaFree(buf_fp32);
	  }

	  if (buf_fp16)
	  {
		cudaFree(buf_fp16);
	  }

	  if (test_inputs_ab)
	  {
		cudaFree(test_inputs_ab);
	  }

	  if (test_inputs_c)
	  {
		cudaFree(&test_inputs_c);
	  }

	  if (result_gpu)
	  {
		cudaFree(result_gpu);
	  }
	  */

	//   free(result_cpu);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
	cudaFree(dD);

	free(hA);
	free(hB);
	free(hC);
	free(hD);

	cudaDeviceSynchronize();
}