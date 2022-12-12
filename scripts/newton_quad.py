from galle.sycl import *
from galle.gen_base import *

from galle.gen_particle.kernel_symbols import *
from galle.gen_particle.loops import *

import numpy as np
import inspect
import ast

import pymbolic as pmbl
import pymbolic.primitives as p

from pymbolic.mapper.c_code import CCodeMapper as CCM
from pymbolic.mapper.differentiator import DifferentiationMapper as DM


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



x0p1 = p.CommonSubexpression(1.0 + x0, prefix="x0p1")
x0m1 = p.CommonSubexpression(1.0 - x0, prefix="x0m1")
x1p1 = p.CommonSubexpression(1.0 + x1, prefix="x1p1")
x1m1 = p.CommonSubexpression(1.0 - x1, prefix="x1m1")

f0 = v00 * x0m1 * x1m1 + \
     v10 * x0p1 * x1m1 + \
     v20 * x0m1 * x1p1 + \
     v30 * x0p1 * x1p1

f1 = v01 * x0m1 * x1m1 + \
     v11 * x0p1 * x1m1 + \
     v21 * x0m1 * x1p1 + \
     v31 * x0p1 * x1p1


J00 = p.CommonSubexpression(DM(x0)(f0), prefix="J00")
J01 = p.CommonSubexpression(DM(x1)(f0), prefix="J01")
J10 = p.CommonSubexpression(DM(x0)(f1), prefix="J10")
J11 = p.CommonSubexpression(DM(x1)(f1), prefix="J11")


L00 = 1.0
L10 = J01 / J00
L11 = 1.0

U00 = J00
U01 = J01
U11 = J11 - (J01*J01)/J00

b0 = -1.0 * f0
b1 = -1.0 * f1

a1 = p.CommonSubexpression((1.0/U11) * (-L10 * b0 + b1), prefix="a1")
a0 = p.CommonSubexpression((1.0/U00)*(b0 - U01*a1), prefix="a0")

xnp10 = a0 + x0;
xnp11 = a1 + x1;

ccm = CCM()
r0 = ccm(xnp10)
r1 = ccm(xnp11)

for name, value in ccm.cse_name_list:
    print("const double %s = %s;" % (name, value))

print("x0 = {};".format(r0))
print("x1 = {};".format(r1))

