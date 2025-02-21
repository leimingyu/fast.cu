#!/bin/bash

# Directory containing test files
TEST_DIR="./feb14/e4m3"

# Check if the directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory $TEST_DIR not found!"
    exit 1
fi

# Specify which GPU to use (0, 1, etc.)
gpu_dev=0

# # Create output directory if it doesn't exist
# mkdir -p out

# # Build the program
# make matmul

# Iterate through all txt files in the directory
for test_file in "$TEST_DIR"/*.txt; do
    # Get just the filename without path
    filename=$(basename "$test_file")
    
    # Run test for e4m3
    echo "Running e4m3 test for $filename..."
    if ./out/matmul "$test_file" e4m3 ${gpu_dev}; then
        if [ -f "gpu_output.txt" ]; then
            mv gpu_output.txt "${filename%.*}-result-e4m3.txt"
            echo "Successfully created ${filename%.*}-result-e4m3.txt"
        else
            echo "Warning: gpu_output.txt was not created for e4m3 test"
        fi
    else
        echo "Error: e4m3 test failed for $filename"
    fi
    
    echo "Completed tests for $filename"
    echo "----------------------------"
done


echo "All tests completed!"
