#!/bin/bash

# Directory containing test files
TEST_DIR="./random"

# Check if the directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory $TEST_DIR not found!"
    exit 1
fi

# Specify which GPU to use (0, 1, etc.)
gpu_dev=0

# Iterate through all txt files in the directory
for test_file in "$TEST_DIR"/*.txt; do
    # Get just the filename without path
    filename=$(basename "$test_file")
    
    # Run test for e5m2
    echo "Running fp16 test for $filename..."
    if ./out/matmul-fp16in-fp16out "$test_file" ${gpu_dev}; then
        if [ -f "gpu_output.txt" ]; then
            mv gpu_output.txt "${filename%.*}-result-fp16.txt"
            echo "Successfully created ${filename%.*}-result-fp16.txt"
        else
            echo "Warning: gpu_output.txt was not created for fp16 test"
        fi
    else
        echo "Error: wgmma fp16 test failed for $filename"
    fi
    
    echo "Completed tests for $filename"
    echo "----------------------------"
done

echo "All tests completed!"
