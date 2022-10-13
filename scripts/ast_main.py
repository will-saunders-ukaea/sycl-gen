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
def bar(x):
    return x + four_two

def foo(x,y):
    return x+1, bar(y)+2





def k_call(P, V):
    P[px, 1] = dt
    #P[px, 1] = 1 + bar(V[px, 2])
    a,b = foo(P[px,1], V[px,0])
    #c,d = x,y


Loop(
    # k,
    k_conditional,
    #k_call,
    P,
    V,
)
