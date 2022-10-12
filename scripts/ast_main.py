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

def k(P,V):
    for dimx in range(ndim):
        P[px,dimx] = P[px, dimx] + dt * V[px, dimx]


Loop(
    k,
    P,V
)






