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



four_two = 42

@kernel_inline
def bar(x):
    x = x * 2
    return x + four_two

@kernel_inline
def foo(x,y):
    return bar(y) + x

@kernel_inline
def branch(x):
    if x < 1:
        return 1
    else:
        return 2



@kernel
def k_call(P, V):
    
    P[px,1] = branch(V[px, 2])



Loop(
    # k,
    #k_conditional,
    k_call,
    P,
    V,
)
