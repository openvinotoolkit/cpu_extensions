# About CPU_Extensions
CPU_Extensions is a compute library containing processor optimized kernels code.

# Unit tests for CPU_Extensions
## Tests for kernels
Tests for kernels are written in gtest under tests\src, use ./cpu_extensions_tests to run it.

## Tests for complex features
Some features have many steps and the reference could not be easily written using gtest. For these features can use python to generate the reference. The directory tests\script contains these test, please refer [test in python](./tests/script/README.md).