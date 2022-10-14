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

        k_ast = self.kernel.ast
        kernel_params = k_ast.body[0].args.args

        if len(kernel_params) != len(self.args):
            raise RuntimeError("Number of kernel arguments does not match number of loop arguments.")

        deps = get_dependencies(self.kernel)
        kernel_args = {}
        for vi, varx in enumerate(kernel_params):
            kernel_args[varx.arg] = self.args[vi]

        # correct the ParticleDat access
        particle_dat_rewrite = ParticleDatReWrite(kernel_args, deps["node_globals"])
        deps["func_ast"] = particle_dat_rewrite.visit(deps["node_ast"])

        function_rewrite_constants(deps)
        print("-" * 60)
        inline_functions(deps)

        # print(ast.dump(deps["node_ast"], indent=2))

        visitor = GalleVisitor(self.args, deps["node_globals"])
        visitor.visit(deps["node_ast"])

        for nx in visitor.body_nodes:
            print(nx)
