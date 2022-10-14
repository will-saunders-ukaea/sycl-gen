import inspect
import ast

import pymbolic as pmbl
import cgen


class KernelWritable:
    def reset_writes(self):
        self.gen_writes = {}


class KernelReadable:
    def reset_reads(self):
        self.gen_reads = {}


class Variable(pmbl.primitives.Variable):
    pass


class Constant:
    def __init__(self, value):
        self.value = value


class KernelSymbol(pmbl.primitives.Variable):
    def __init__(self, obj, name=None):
        pmbl.primitives.Variable.__init__(self, name)


class ParticleSymbol(KernelSymbol, KernelWritable, KernelReadable):
    def __init__(self, obj, name=None):
        KernelSymbol.__init__(self, obj, name)

    def get_access(self, *args):
        symbol = args[0]
        cellx = args[1]
        layerx = args[2]
        component = args[3]
        return Variable(symbol)[cellx][component][layerx]

    def __getitem__(self, *args):
        particle_loop = args[0][0]
        cellx = particle_loop.gen_loop_cell
        layerx = particle_loop.gen_loop_layer
        component = args[0][1]
        return self.get_access(self.name, cellx, layerx, component)


class LocalArray:
    def __init__(self, ncomp, dtype="REAL"):
        self.ncomp = ncomp
        self.dtype = dtype

    def rename(self, name):
        self.name = name


class KernelFunction:
    def __init__(self, func):
        self.func = func
        self.globals = inspect.getclosurevars(func).globals
        self.ast = ast.parse(inspect.getsource(func))


def kernel_inline(func):
    return KernelFunction(func)


def kernel(func):
    return KernelFunction(func)
