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
dt = 0.001


ndim = Constant(2)


def foo(x):
    return x + 1


def k(P, V):
    for dimx in range(ndim):
        tmp = dt * V[px, dimx]
        tmp = foo(tmp)
        P[px, dimx] = P[px, dimx] + tmp


def k_conditional(P, V):

    v = V[px, 0]
    if v == 1:
        P[px, 0] = 0
    elif v == 2:
        P[px, 1] = 1
    else:
        P[px, 2] = 2


Loop(
    # k,
    k_conditional,
    P,
    V,
)
