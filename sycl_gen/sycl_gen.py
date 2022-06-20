import ctypes
import os
import numpy as np
import shlex
import subprocess
import hashlib

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

        with open(filename_source, "w") as fh:
            fh.write('#include "' + h + '.hpp"\n')
            fh.write(source)

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
        """
        lib_src = """
        """

        self.lib = compiler(lib_header, lib_src, "sycl_lib_")
