from galle.sycl import *
from galle.gen_base import *


from galle.gen_particle.kernel_symbols import *
from galle.gen_particle.loops import *


import numpy as np
import inspect
import ast


px = ParticleLoop()

P = ParticleSymbol(None, "P")
V = ParticleSymbol(None, "V")
B = ParticleSymbol(None, "B")
E = ParticleSymbol(None, "E")
dt = 0.001


ndim = Constant(2)


@kernel_inline
def cross_product_3d(a1, a2, a3, b1, b2, b3):
    c1 = (a2 * b3) - (a3 * b2)
    c2 = (a3 * b1) - (a1 * b3)
    c3 = (a1 * b2) - (a2 * b1)
    return c1, c2, c3


@kernel_inline
def dot_product_3d(a1, a2, a3, b1, b2, b3):
    return (a1 * b1) + (a2 * b2) + (a3 * b3)


@kernel_inline
def l2_squared_3d(a1, a2, a3):
    return dot_product_3d(a1, a2, a3, a1, a2, a3)


q = Constant(1.0)
m = Constant(1.0)


@kernel
def k_boris(V, B, E, t):
    scaling_t = (q / m) * 0.5 * dt

    t[0] = 1


Loop(k_boris, V, B, E, LocalArray(3))
