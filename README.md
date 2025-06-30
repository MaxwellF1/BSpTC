# Bullseye Hash

Bullseye Hash is a high performance hash table to accelerate element-wise sparse tensor contraction on multicore CPU. Bullseye Hash is implemented based on the open-sourced Hierarchical Parallel Tensor Infrastructure (HiParTI) library.

## Environments

- GCC
- CMake
- OpenBLAS
- NUMA

## Setup
```
git clone xxx (Bullseye Hash)
git clone xxx (Tensors)
```

./build.sh

And you can set the running modes (including Bullseye Hash and other implementations such as seperate chaining, linear probing and cuckoo hash, as well as the COO version)

you can run the test scripts and change the parameters
./scripts/run.sh