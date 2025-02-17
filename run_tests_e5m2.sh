#!/bin/bash

# Directory containing test files
TEST_DIR="./feb14/e5m2"

# Check if the directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory $TEST_DIR not found!"
    exit 1
fi

# Iterate through all txt files in the directory
for test_file in "$TEST_DIR"/*.txt; do
    # Get just the filename without path
    filename=$(basename "$test_file")
    
    
    # Run test for e5m2
    echo "Running e5m2 test for $filename..."
    if ./out/matmul "$test_file" e5m2; then
        if [ -f "gpu_output.txt" ]; then
            mv gpu_output.txt "${filename%.*}-result-e5m2.txt"
            echo "Successfully created ${filename%.*}-result-e5m2.txt"
        else
            echo "Warning: gpu_output.txt was not created for e5m2 test"
        fi
    else
        echo "Error: e5m2 test failed for $filename"
    fi
    
    echo "Completed tests for $filename"
    echo "----------------------------"
done

echo "All tests completed!"
