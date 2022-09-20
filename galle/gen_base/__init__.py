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

        self.lib_src = """
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
        
        self.lib = sycl.compiler(lib_header, self.lib_src)["wrapper"]

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
