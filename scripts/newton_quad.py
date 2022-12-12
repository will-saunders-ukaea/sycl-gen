from galle.sycl import *
from galle.gen_base import *

from galle.gen_particle.kernel_symbols import *
from galle.gen_particle.loops import *

import numpy as np
import inspect
import ast

import pymbolic as pmbl
import pymbolic.primitives as p


from pymbolic.mapper.differentiator import DifferentiationMapper as DM
print()


x0 = p.Variable("x0")
x1 = p.Variable("x1")

v00 = p.Variable("v00")
v01 = p.Variable("v01")
v10 = p.Variable("v10")
v11 = p.Variable("v11")
v20 = p.Variable("v20")
v21 = p.Variable("v21")
v30 = p.Variable("v30")
v31 = p.Variable("v31")

f0 = v00 * (1.0 - x0) * (1.0 - x1) + \
     v10 * (1.0 + x0) * (1.0 - x1) + \
     v20 * (1.0 - x0) * (1.0 + x1) + \
     v30 * (1.0 + x0) * (1.0 + x1)
f1 = v01 * (1.0 - x0) * (1.0 - x1) + \
     v11 * (1.0 + x0) * (1.0 - x1) + \
     v21 * (1.0 - x0) * (1.0 + x1) + \
     v31 * (1.0 + x0) * (1.0 + x1)


J00 = p.CommonSubexpression(DM(x0)(f0))
J01 = p.CommonSubexpression(DM(x1)(f0))
J10 = p.CommonSubexpression(DM(x0)(f1))
J11 = p.CommonSubexpression(DM(x1)(f1))


L00 = 1.0
L10 = J01 / J00
L11 = 1.0

U00 = J00
U01 = J01
U11 = J11 - (J01*J01)/J00

b0 = p.Variable("b0")
b1 = p.Variable("b1")

a1 = p.CommonSubexpression((1.0/U11) * (-L10 * b0 + b1))
a0 = p.CommonSubexpression((1.0/U00)*(b0 - U01*a1))

print(a0)
print(a1)





