#!/bin/bash

# Directory containing test files
TEST_DIR="./feb7"

# Check if the directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory $TEST_DIR not found!"
    exit 1
fi

# Iterate through all txt files in the directory
for test_file in "$TEST_DIR"/*.txt; do
    # Get just the filename without path
    filename=$(basename "$test_file")
    
    # Run test for e4m3
    echo "Running e4m3 test for $filename..."
    ./out/matmul "$test_file" e4m3

    mv gpu_output.txt "${filename%.*}-result-e4m3.txt"
    
    # Run test for e5m2
    echo "Running e5m2 test for $filename..."
    ./out/matmul "$test_file" e5m2
    mv gpu_output.txt "${filename%.*}-result-e5m2.txt"
    
    echo "Completed tests for $filename"
    echo "----------------------------"
done

echo "All tests completed!"
