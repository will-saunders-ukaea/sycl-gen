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
def k_boris(V, B, E, t, s, V_minus, V_prime, V_plus):

    scaling_t = (q / m) * 0.5 * dt
    for dimx in range(3):
        t[dimx] = B[px, dimx] * scaling_t

    tmagsq = l2_squared_3d(t0, t1, t2)
    scaling_s = 2.0 / (1.0 + tmagsq)

    for dimx in range(3):
        s[dimx] = scaling_s * t[dimx]
        V_minus[dimx] = V[px, dimx] + E[px, dimx] * scaling_t

    v_prime[0], v_prime[1], v_prime[2] = cross_product_3d(v_minus[0], v_minus[1], v_minus[2], t[0], t[1], t[2])

    for dimx in range(3):
        v_prime[dimx] = v_prime[dimx] + v_minus[dimx]

    v_plus[0], v_plus[1], v_plus[2] = cross_product_3d(v_prime[0], v_prime[1], v_prime[2], s[0], s[1], s[2])

    for dimx in range(3):
        v_plus[dimx] = v_plus[dimx] + v_minus[dimx]
        V[px, dimx] = v_plus[dimx] + scaling_t * E[px, dimx]


Loop(
    k_boris,
    V,
    B,
    E,
    LocalArray(3),
    LocalArray(3),
    LocalArray(3),
    LocalArray(3),
    LocalArray(3),
)
