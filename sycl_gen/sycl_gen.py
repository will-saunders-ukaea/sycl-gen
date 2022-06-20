import ctypes
import os
import numpy as np
import shlex
import subprocess
import hashlib
from functools import reduce
from collections.abc import Iterable

REAL = ctypes.c_double
INT = ctypes.c_int64


class Compiler:
    def __init__(self, binary, args, build_dir="./build"):
        self.binary = binary
        self.args = shlex.split(args)
        self.build_dir = os.path.abspath(build_dir)
        if not os.path.exists(self.build_dir):
            os.mkdir(self.build_dir)

    def __call__(self, header, source, name=""):

        m = hashlib.sha256()
        m.update((header + source + name + self.binary + "".join(self.args)).encode("utf-8"))
        h = name + m.hexdigest()

        filename_header = os.path.join(self.build_dir, h + ".hpp")
        filename_source = os.path.join(self.build_dir, h + ".cpp")
        filename_lib = os.path.join(self.build_dir, h + ".so")

        with open(filename_header, "w") as fh:
            fh.write(header)
            fh.write("\n")

        with open(filename_source, "w") as fh:
            fh.write('#include "' + h + '.hpp"\n')
            fh.write(source)
            fh.write("\n")

        if not os.path.exists(filename_lib):
            cmd = (
                [
                    self.binary,
                ]
                + [
                    filename_source,
                ]
                + self.args
                + ["-o", filename_lib]
            )

            subprocess.check_call(cmd)

        lib = ctypes.cdll.LoadLibrary(filename_lib)

        return lib


class SYCL:
    def __init__(self, compiler):
        self.compiler = compiler

        lib_header = """
        #include <CL/sycl.hpp>
        #include <iostream>
        """
        lib_src = """
        
        using namespace cl;
        using namespace cl::sycl;

        extern "C" int gen_initialise(sycl::device ** d, sycl::queue ** q){
            *d = new sycl::device(sycl::default_selector());
            *q = new sycl::queue(**d);
            return 0;
        }
        extern "C" int gen_print_device(sycl::device *d, sycl::queue *q){
            //std::cout << "Using " << d->get_info<sycl::info::device::name>()
            //          << std::endl;
            auto d2 = q->get_device();
            std::cout << "Using " << d2.get_info<sycl::info::device::name>()
                      << std::endl;
            return 0;
        }
        extern "C" int gen_finalise(sycl::device ** d, sycl::queue ** q){
            delete *q;
            delete *d;
            return 0;
        }
        extern "C" int gen_malloc_shared(size_t num_bytes, sycl::queue *q, char ** ptr){
            *ptr = (char *) sycl::malloc_shared(num_bytes, *q);
            return 0;
        }
        extern "C" int gen_sycl_free(sycl::queue *q, char *ptr){
            sycl::free(ptr, *q);
            return 0;
        }

        """

        self.lib = compiler(lib_header, lib_src, "sycl_lib_")

        self.device = ctypes.c_void_p(0)
        self.queue = ctypes.c_void_p(0)

        self.lib["gen_initialise"](ctypes.byref(self.device), ctypes.byref(self.queue))
        self.lib["gen_print_device"](self.device, self.queue)

    def __del__(self):

        self.lib["gen_finalise"](ctypes.byref(self.device), ctypes.byref(self.queue))

    def zeros_shared(self, shape, dtype):

        if not issubclass(type(shape), Iterable):
            shape = (shape,)
        num_bytes = reduce(lambda x, y: x * y, shape) * ctypes.sizeof(dtype)

        ptr = ctypes.c_void_p()

        self.lib["gen_malloc_shared"](ctypes.c_size_t(num_bytes), self.queue, ctypes.byref(ptr))

        typed_ptr = ctypes.cast(ptr, ctypes.POINTER(dtype))
        array = np.ctypeslib.as_array(typed_ptr, shape=shape)

        return array

    def free(self, array):

        ptr = ctypes.c_void_p(array.ctypes.data)

        self.lib["gen_sycl_free"](self.queue, ptr)
