from sycl_gen import *

# syclcc = Compiler("syclcc", "--hipsycl-targets=omp -O3 -fPIC -shared")
syclcc = Compiler("syclcc", "--hipsycl-targets=cuda-nvcxx -O3 -fPIC -shared")

sycl = SYCL(syclcc)

N = 1024
a_array = sycl.zeros_shared((N,), REAL)
b_array = sycl.zeros_shared((N,), REAL)
c_array = sycl.zeros_shared((N,), REAL)


sycl.free(a_array)
sycl.free(b_array)
sycl.free(c_array)
