import inspect
import ast

import pymbolic as pmbl
from pymbolic.mapper.c_code import CCodeMapper as CCM

from galle.gen_ir.ast_cgen import *
from galle.gen_particle.kernel_symbols import Variable

from galle.gen_particle.kernel_symbols import *
from galle.gen_particle.ast_transforms import *


class ParticleIterator(Variable):
    pass


class ParticleLoop(ParticleIterator):
    def __init__(self):
        super().__init__(name="ParticleLoop")
        self.sym_loop_cell = "neso_cellx"
        self.sym_loop_layer = "neso_layerx"
        self.gen_loop_cell = Variable(name="neso_cellx")
        self.gen_loop_layer = Variable(name="neso_layerx")

    def get_access(self):
        return [self.gen_loop_cell, self.gen_loop_layer]


class Loop:
    def __init__(self, *args, **kwargs):

        self.kernel = args[0]
        self.args = args[1:]

        print(self.kernel)
        print(self.args)

        kernel_vars = inspect.getclosurevars(self.kernel)
        k_ast = ast.parse(inspect.getsource(self.kernel))

        kernel_params = k_ast.body[0].args.args

        print(kernel_vars.globals)

        kernel_args = {}
        for vi, varx in enumerate(kernel_params):
            kernel_args[varx.arg] = self.args[vi]

        # correct the ParticleDat access
        particle_dat_rewrite = ParticleDatReWrite(kernel_args, kernel_vars.globals)
        k_ast = particle_dat_rewrite.visit(k_ast)

        # replace constants with values
        constants_rewrite = ConstantReWrite(kernel_vars.globals)
        k_ast = constants_rewrite.visit(k_ast)

        print(ast.dump(k_ast, indent=2))

        visitor = GalleVisitor(self.args, kernel_vars.globals)
        visitor.visit(k_ast)

        for nx in visitor.body_nodes:
            print(nx)
