from sycl_gen import *

syclcc = Compiler("syclcc", "-c --hipsycl-targets=omp -O3")

sycl = SYCL(syclcc)
