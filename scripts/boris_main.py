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
def k_boris(V, B, E):

    scaling_t = (q / m) * 0.5 * dt
    t0 = B[px, 0] * scaling_t
    t1 = B[px, 1] * scaling_t
    t2 = B[px, 2] * scaling_t

    tmagsq = l2_squared_3d(t0, t1, t2)
    scaling_s = 2.0 / (1.0 + tmagsq)

    s0 = scaling_s * t[0]
    s1 = scaling_s * t[1]
    s2 = scaling_s * t[2]

    V_minus_0 = V[px, 0] + E[px, 0] * scaling_t
    V_minus_1 = V[px, 1] + E[px, 1] * scaling_t
    V_minus_2 = V[px, 2] + E[px, 2] * scaling_t

    v_prime_0, v_prime_1, v_prime_2 = cross_product_3d(v_minus_0, v_minus_1, v_minus_2, t0, t1, t2)

    v_prime_0 = v_prime_0 + v_minus_0
    v_prime_1 = v_prime_1 + v_minus_1
    v_prime_2 = v_prime_2 + v_minus_2

    v_plus_0, v_plus_1, v_plus_2 = cross_product_3d(v_prime_0, v_prime_1, v_prime_2, s0, s1, s2)

    v_plus_0 = v_plus_0 + v_minus_0
    v_plus_1 = v_plus_1 + v_minus_1
    v_plus_2 = v_plus_2 + v_minus_2

    V[px, 0] = v_plus_0 + scaling_t * E[px, 0]
    V[px, 1] = v_plus_1 + scaling_t * E[px, 1]
    V[px, 2] = v_plus_2 + scaling_t * E[px, 2]


Loop(
    k_boris,
    V,
    B,
    E,
)
