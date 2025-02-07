# About

This project is about testing the results of fp8in-fp16out using wgmma api.

Code is based on [`matmul.cu`](https://github.com/pranjalssh/fast.cu/blob/main/matmul.cu).


## Build and Run (Ubuntu):
```
make matmul && out/matmul <test_input.txt> <fp8format:e5m2/e4m3>
```
