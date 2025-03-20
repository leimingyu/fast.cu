//----------------------------------------------------------------------------//
// 5th gen tensor core for fp8 on RTX 5080 (blackwell, sm120)
//
// ptx :    tcgen05.mma.cta_group::1.kind::f8f6f4
// tile:    M64N8K32
//
// more details : https://confluence.qualcomm.com/confluence/display/TENSOR/Microscaling+for+FP8
//----------------------------------------------------------------------------//


#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <iomanip> // setfill, setw
#include <cstdint> // int8_t, uint8_t


#include <cuda.h>
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_fp8.h> // FP8


#define DEBUG 1


#define K32 32


enum class FP8Format
{
    E5M2,
    E4M3
};


//----------------------------------------------------------------------------//
// Utility
//----------------------------------------------------------------------------//
void logMessage(const char *format, ...)
{
    // Get the current timestamp
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));


    // Print the timestamp
    std::cout << "[" << timestamp << "] ";


    // Handle the variadic arguments
    va_list args;
    va_start(args, format);
    vprintf(format, args); // Print the formatted message
    va_end(args);


    // End with a newline
    std::cout << std::endl;
}


void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(1);
    }
}


#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))


//----------------------------------------------------------------------------//
// Functions
//----------------------------------------------------------------------------//
void printProgressBar(double percentage);


template <int TILE_K, FP8Format FORMAT>
void runTest(std::vector<std::vector<uint8_t>> &current_test_ab,
             std::vector<uint32_t> &current_test_c,
             std::vector<uint32_t> &current_result,
             int SM_num);


//----------------------------------------------------------------------------//
// gpu kernel
//----------------------------------------------------------------------------//


//----------------------------------------------------------------------------//
//
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
    if (format_str == "e5m2")
    {
        use_e5m2 = true;
    }
    else if (format_str == "e4m3")
    {
        use_e5m2 = false;
    }
    else
    {
        std::cerr << "Error: format must be either 'e5m2' or 'e4m3'" << std::endl;
        return 1;
    }


    // Parse and set the device ID
    int device_id = std::stoi(argv[3]);
    int device_count;
    cudaCheck(cudaGetDeviceCount(&device_count));


    if (device_id < 0 || device_id >= device_count)
    {
        std::cerr << "Error: device_id must be between 0 and " << (device_count - 1) << std::endl;
        return 1;
    }


    cudaCheck(cudaSetDevice(device_id));


    // Get and print device properties
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, device_id));
    logMessage("Using GPU device %d: %s", device_id, prop.name);
    logMessage("Number of SMs: %d\n", prop.multiProcessorCount);


    int SM_NUM = prop.multiProcessorCount;


    //------------------------------------------------------------------------//
    // Read all test cases
    //------------------------------------------------------------------------//
    // 65 input values per row:   c +  32 of a/b for K32 case
    // 33 input values per row:   c +  16 of a/b for K16 case
    // 17 input values per row:   c +   8 of a/b for  K8 case
    std::vector<uint32_t> allTests_c;              // input c in fp32
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
    std::streamsize totalRead = 0;
    size_t lineCount = 0;


    while (getline(file, line))
    {
        // print progress
        lineCount++;
        totalRead += line.size() + 1;
        if (lineCount % 100 == 0)
            printProgressBar(static_cast<double>(totalRead) / fileSize);


        // read line :    c a0 b0 a1 b1 ....  aN bN
        std::istringstream iss(line);
        std::vector<uint8_t> numbers_ab; // current line for a/b
        std::string hexStr;


        // read c first
        iss >> hexStr;
        uint32_t num_c = static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16)); // was uint16_t
        allTests_c.push_back(num_c);                                             // store current line for C


        // read a/b
        while (iss >> hexStr)
        {
            uint8_t num = static_cast<uint8_t>(std::stoul(hexStr, nullptr, 16));
            numbers_ab.push_back(num);
        }


        // Check if this is a K8/K16/K32 case
        if (numbers_ab.size() == 32) // K16
        {
            numbers_ab.resize(64, 0); // Add 32 zeros
        }
        else if (numbers_ab.size() == 16)
        {
            numbers_ab.resize(64, 0); // Add zeros
        }
        else if (numbers_ab.size() != 64)
        {
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
    printf("\nCheck first line of input file:\n");
    printf("%08X ", allTests_c[0]);
    for (int i = 0; i < 64; i++)
    {
        printf("%02X ", allTests_ab[0][i]);
    }
    printf("\n");


    //------------------------------------------------------------------------//
    // Run all test cases
    //------------------------------------------------------------------------//
    std::cout << "Run Tensor Core Tests with " << format_str << " format\n"
              << std::endl;


    // Change from int to size_t for totalNum
    size_t totalNum = allTests_ab.size();


    // results in fp32
    std::vector<uint32_t> allTests_results(totalNum);


    size_t batches = (totalNum + SM_NUM - 1) / SM_NUM;


    for (size_t i = 0; i < batches; ++i)
    {
        size_t start_idx = i * SM_NUM;
        size_t end_idx = std::min(start_idx + SM_NUM, totalNum);


        if ((end_idx % 1000) == 0)
            logMessage("case : %zu (%zu : %.2f %% done) \n", end_idx, totalNum, (end_idx / (float)totalNum) * 100);


        size_t test_counts = end_idx - start_idx;


        // Create tests for this batch from allTests_c and allTests_ab.
        std::vector<uint32_t> partTests_c(allTests_c.begin() + start_idx, allTests_c.begin() + end_idx);
        std::vector<std::vector<uint8_t>> partTests_ab(allTests_ab.begin() + start_idx, allTests_ab.begin() + end_idx);


        // output in fp32
        std::vector<uint32_t> current_result(test_counts);


        //--------------------------------------------------------------------//
        // run tensor core test
        //--------------------------------------------------------------------//
        if (use_e5m2)
        { // Add a command line argument or configuration to set this
            runTest<K32, FP8Format::E5M2>(partTests_ab, partTests_c, current_result, SM_NUM);
        }
        else
        {
            // runTest<K32, FP8Format::E4M3>(partTests_ab, partTests_c, current_result);
        }


        // update the test results
        std::copy(current_result.begin(), current_result.end(), allTests_results.begin() + start_idx);
    }


    //------------------------------------------------------------------------//
    // Export the results
    //------------------------------------------------------------------------//
    // std::string outFileName = "gpu_output.txt";
    // std::ofstream outFile(outFileName);
    // if (!outFile)
    // {
    //     std::cerr << "Error opening file to write." << std::endl;
    //     return 1;
    // }


    // const int colNum = 2;
    // for (int i = 0; i < totalNum; i++)
    // {
    //     for (int j = 0; j < colNum; j++)
    //     {
    //         // outFile << allTests_results[i][j];
    //         outFile << std::setfill('0') << std::setw(8) << std::hex << allTests_results[i][j];
    //         if (j < colNum)
    //         {
    //             outFile << " "; // separate with space
    //         }
    //     }
    //     outFile << "\n"; // EOR
    // }


    // outFile.close();


    // std::cout << "\nResults are saved! Check " << outFileName << ".\n";


    return 0;
}


//----------------------------------------------------------------------------//
// print read input progress
//----------------------------------------------------------------------------//
void printProgressBar(double percentage)
{
    int barWidth = 50;
    std::cout << "[";
    int pos = static_cast<int>(barWidth * percentage);
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << " %\r";
    std::cout.flush();
}


//----------------------------------------------------------------------------//
// host code to prepare test
//----------------------------------------------------------------------------//
template <int TILE_K, FP8Format FORMAT>
void runTest(std::vector<std::vector<uint8_t>> &current_test_ab,
             std::vector<uint32_t> &current_test_c,
             std::vector<uint32_t> &current_result,
             int SM_num)
{
    int test_batch_size = current_test_ab.size();

    std::cout << "test size :  " << test_batch_size << std::endl;
    std::cout << "using SM  =   " << SM_num << std::endl;


    //------------------------------------------------------------------------//
    // mma size :  M 64 x N 8 x K 32
    //------------------------------------------------------------------------//
    long M = 64, N = 8, K = 32;


    //------------------------------------------------------------------------//
    // prepare host buffers
    //------------------------------------------------------------------------//
    uint8_t *hA = nullptr;
    uint8_t *hB = nullptr;

    uint32_t *hD = nullptr;
    uint32_t *hresult = nullptr;

    size_t sizeA = M * K * SM_num;
    size_t sizeB = N * K * SM_num;
    size_t sizeD = M * N * SM_num;   // noted: for fp16 accumulation, each 32 reg will store two fp16 results

    hA = (uint8_t *)malloc(sizeof(uint8_t) * sizeA);
    hB = (uint8_t *)malloc(sizeof(uint8_t) * sizeB);
    hD = (uint32_t *)malloc(sizeof(uint32_t) * sizeD);
    hresult = (uint32_t *)malloc(sizeof(uint32_t) * sizeD);

    // init to 0
    memset(hA, 0, sizeof(uint8_t) * sizeA);     // MxKxSM_num
    memset(hB, 0, sizeof(uint8_t) * sizeB);     // NxKxSM_num
    memset(hD, 0, sizeof(uint32_t) * sizeD);    // MxNxSM_num
    memset(hresult, 0, sizeof(uint32_t) * sizeD);


    //------------------------------------------------------------------------//
    // read/set up data on cpu
    //------------------------------------------------------------------------//
    // read a/b for each test case in the batch
    for (int test = 0; test < test_batch_size; test++) {

        // Calculate base offsets for this test case
        size_t a_offset = test * M * K;  // Offset in hA array
        size_t b_offset = test * N * K;  // Offset in hB array
        size_t d_offset = test * M * N;  // Offset in hD array

        // Read a/b pairs for this test case
        for (int i = 0; i < TILE_K; i++) {
            // MxK: read 'a' values
            hA[a_offset + i] = current_test_ab[test][i * 2];
            // NxK: read 'b' values
            hB[b_offset + i] = current_test_ab[test][i * 2 + 1];
        }

        // Read input C for this test case:  1st element of hD, with step size of MxN
        hD[d_offset] = (uint32_t)current_test_c[test];
    }




// #if DEBUG
//     std::cout << "Read input C" << std::endl;
// #endif


// #if DEBUG
//     printf("Pack input C (fp32) : %08X \n\n", hD[0]);
// #endif




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
    //matmul_fp8_64x8x32_kernel<FORMAT><<<1, 128>>>(dA, dB, dD);


    // d2h : copy results back to host
    cudaMemcpy(hresult, dD, sizeof(uint32_t) * sizeD, cudaMemcpyDeviceToHost);

    // note: read the 1st element of each MxN block

    /*
    uint32_t c0 = hresult[0];
    // uint16_t c0_lo = static_cast<uint16_t>(c0 & 0xFFFF);   // read the lower half (1st 16 bits)
    // uint16_t c0_hi = static_cast<uint16_t>((c0 >> 16) & 0xFFFF);


// check value
#if DEBUG
    printf("%08X\n", hresult[0]);
#endif


    //printf("\n\n");


    current_result.push_back(c0);


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);


    free(hA);
    free(hB);
    free(hD);
    free(hresult);


    cudaDeviceSynchronize();
    */
}
