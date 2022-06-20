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


class KernelSymbol(pmbl.primitives.Variable):

    name_counter = 0
    name_base = "sym_"

    def __init__(self, obj, name=None):
        if name is None:
            name = KernelSymbol.name_base + str(KernelSymbol.name_counter)
            KernelSymbol.name_counter += 1
        super().__init__(name)
        self.gen_obj = obj


class NDRange(pmbl.primitives.Variable):
    def __init__(self, n, name="idx"):
        super().__init__(name)
        self.gen_n = n


class Assign:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.op = "="


class Executor:
    def __init__(self, sycl, iteration_set, *exprs, **kwargs):

        self.sycl = sycl

        lib_header = """
        #include <CL/sycl.hpp>
        #include <iostream>
        #include <cstdint>
        """

        dependency_mapper = pmbl.mapper.dependency.DependencyMapper(
            include_subscripts=False,
            include_lookups=False,
            include_calls=False,
            include_cses=False,
            composite_leaves=None,
        )

        self.parameters = OrderedDict()
        expressions = []

        def process_dependency(dep):
            if issubclass(type(dep), KernelSymbol):
                self.parameters[dep.name] = dep.gen_obj

        ccm = CCM()
        for expr in exprs:
            expressions.append(ccm(expr.lhs) + " " + expr.op + " " + ccm(expr.rhs) + ";")

            for dx in dependency_mapper(expr.lhs):
                process_dependency(dx)
            for dx in dependency_mapper(expr.rhs):
                process_dependency(dx)

        lib_src = """
        using namespace cl;
        using namespace cl::sycl;
        extern "C" int wrapper(
            sycl::queue *queue,
            {PARAMETERS}
        ){{
              queue->submit([&](sycl::handler &cgh) {{
               cgh.parallel_for<>(sycl::range<1>({LOOP_EXTENT}), [=](sycl::id<1> {LOOP_INDEX}) {{
                {EXPRESSIONS}
              }});
            }}).wait();


            return 0;
        }}

        """.format(
            LOOP_EXTENT=str(iteration_set.gen_n),
            LOOP_INDEX=iteration_set.name,
            EXPRESSIONS="\n".join(expressions),
            PARAMETERS=",\n".join([self.format_parameters(fx) for fx in self.parameters.items()]),
        )

        self.lib = sycl.compiler(lib_header, lib_src)["wrapper"]

    def format_parameters(self, fx):
        ctype_map = {REAL: "double", INT: "int64_t"}

        param = "{CTYPE} * {NAME}".format(CTYPE=ctype_map[np.ctypeslib.as_ctypes_type(fx[1].dtype)], NAME=fx[0])

        return param

    def get_argument(self, fx):
        return fx[1].ctypes.get_as_parameter()

    def __call__(self):
        args = [
            self.sycl.queue,
        ]
        args += [self.get_argument(fx) for fx in self.parameters.items()]
        self.lib(*args)
