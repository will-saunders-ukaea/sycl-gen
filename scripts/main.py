from galle.sycl import *
from galle.gen_base import *

import numpy as np


# syclcc = Compiler("syclcc", "--hipsycl-targets=omp -O3 -fPIC -shared")
syclcc = Compiler("syclcc", "--hipsycl-targets=cuda-nvcxx -O3 -fPIC -shared")

sycl = SYCL(syclcc)

N = 1024
a_array = sycl.empty_shared((N,), REAL)
b_array = sycl.empty_shared((N,), REAL)
c_array = sycl.empty_shared((N,), REAL)
d_array = sycl.empty_shared((N,), REAL)
e_array = sycl.empty_shared((N,), REAL)


a_array[:] = np.random.uniform(size=(N,))
b_array[:] = np.random.uniform(size=(N,))
c_array[:] = np.random.uniform(size=(N,))

correct_fma = a_array * b_array + c_array
correct_add = a_array + b_array


a = KernelSymbol(a_array)
b = KernelSymbol(b_array)
c = KernelSymbol(c_array)
d = KernelSymbol(d_array)
e = KernelSymbol(e_array)
ix = NDRange(N)


fma = Executor(
    sycl, 
    ix, 
    Assign(d[ix], a[ix] * b[ix] + c[ix]),
    Assign(e[ix], a[ix] + b[ix])
)

fma()

err = np.linalg.norm(d_array - correct_fma, np.inf)
print("Infinity norm error fma:", err)
err = np.linalg.norm(e_array - correct_add, np.inf)
print("Infinity norm error add:", err)

sycl.free(a_array)
sycl.free(b_array)
sycl.free(c_array)
sycl.free(d_array)
sycl.free(e_array)
