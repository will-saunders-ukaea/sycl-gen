import ctypes
import os
import numpy as np
import shlex
import subprocess
import hashlib
from functools import reduce
from collections.abc import Iterable
from collections import OrderedDict

import pymbolic as pmbl
from pymbolic.mapper.c_code import CCodeMapper as CCM

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

    def empty_shared(self, shape, dtype):

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


class IterationSet:
    pass


class Range1D(IterationSet):
    def __init__(self, n, name="_LOOP_INDEX"):
        self.n = n
        self.name = name

    def get_loop_index(self):
        return f"sycl::id<1> {self.name}"

    def get_declaration_loop(self):
        return "sycl::range<1>(_LOOP_EXTENT)"

    def get_declaration_lib(self):
        return "const int64_t _LOOP_EXTENT"

    def get_call_args(self):
        return ctypes.c_int64(self.n)


class Kernel:
    def __init__(self, ccode, cincludes=""):
        self.ccode = ccode
        self.cincludes = cincludes


class KernelArg:
    def __init__(self, obj, name):
        self.obj = obj
        self.name = name


class ParallelFor:
    def __init__(self, *args, **kwargs):
        self.iteration_set = args[0]
        self.args = args[1:-1]
        self.kernel = args[-1]

        assert issubclass(type(self.iteration_set), IterationSet)
        assert issubclass(type(self.kernel), Kernel)

        self.lib_header = (
            """
        #include <CL/sycl.hpp>
        #include <iostream>
        #include <cstdint>
        """
            + self.kernel.cincludes
        )

        self.generated_elements = {}
        self._generate_parameters()
        self._generate_iteration_set()
        self._generate_iteration_index()
        self._generate_kernel()

        self.lib_src = """
        using namespace cl;
        using namespace cl::sycl;
        extern "C" int wrapper(
            sycl::queue *queue,
            {PARAMETERS}
        ){{
              queue->submit([&](sycl::handler &cgh) {{
               cgh.parallel_for<>({ITERATION_SET}, [=]({ITERATION_INDEX}) {{
                {KERNEL}
              }});
            }}).wait();
            return 0;
        }}

        """.format(
            **self.generated_elements
        )

    def _generate_parameters(self):
        p = [self.iteration_set.get_declaration_lib()]
        ctype_map = {REAL: "double", INT: "int64_t"}
        for argx in self.args:
            assert issubclass(type(argx), KernelArg)
            if issubclass(type(argx.obj), np.ndarray):
                argp = "{CTYPE} * {NAME}".format(
                    CTYPE=ctype_map[np.ctypeslib.as_ctypes_type(argx.obj.dtype)], NAME=argx.name
                )
            else:
                argp = argx.obj.get_declaration_lib()
            p.append(argp)
        self.generated_elements["PARAMETERS"] = ",".join(p)

    def _generate_iteration_set(self):
        self.generated_elements["ITERATION_SET"] = self.iteration_set.get_declaration_loop()

    def _generate_iteration_index(self):
        self.generated_elements["ITERATION_INDEX"] = self.iteration_set.get_loop_index()

    def _generate_kernel(self):
        self.generated_elements["KERNEL"] = self.kernel.ccode
