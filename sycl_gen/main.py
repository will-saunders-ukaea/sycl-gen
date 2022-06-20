from sycl_gen import *

import numpy as np


# syclcc = Compiler("syclcc", "--hipsycl-targets=omp -O3 -fPIC -shared")
syclcc = Compiler("syclcc", "--hipsycl-targets=cuda-nvcxx -O3 -fPIC -shared")

sycl = SYCL(syclcc)

N = 1024
a_array = sycl.zeros_shared((N,), REAL)
b_array = sycl.zeros_shared((N,), REAL)
c_array = sycl.zeros_shared((N,), REAL)
d_array = sycl.zeros_shared((N,), REAL)


a_array[:] = np.random.uniform(size=(N,))
b_array[:] = np.random.uniform(size=(N,))
c_array[:] = np.random.uniform(size=(N,))

correct = a_array * b_array + c_array


a = KernelSymbol(a_array)
b = KernelSymbol(b_array)
c = KernelSymbol(c_array)
d = KernelSymbol(d_array)
ix = NDRange(N)


fma = Executor(sycl, ix, Assign(d[ix], a[ix] * b[ix] + c[ix]))


fma()

err = np.linalg.norm(d_array - correct, np.inf)

print("Infinity norm error:", err)

sycl.free(a_array)
sycl.free(b_array)
sycl.free(c_array)
sycl.free(d_array)
