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







def k(P,V):
    P[px,0] = P[px, 0] + dt * V[px, 0]
    P[px,1] = P[px, 1] + dt * V[px, 1]


Loop(
    k,
    P,V
)






