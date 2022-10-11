import inspect
import ast

import pymbolic as pmbl
from pymbolic.mapper.c_code import CCodeMapper as CCM

from galle.gen_ir.ast_cgen import *
from galle.gen_particle.kernel_symbols import Variable

from galle.gen_particle.kernel_symbols import *


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


class ParticleDatReWrite(ast.NodeTransformer):
    def __init__(self, kernel_args, kernel_globals):
        ast.NodeTransformer.__init__(self)
        self.kernel_args = kernel_args
        self.kernel_globals = kernel_globals

    def visit_Subscript(self, node):

        symbol = node.value.id
        if (symbol in self.kernel_args.keys()) and (issubclass(type(self.kernel_args[symbol]), ParticleSymbol)):

            particle_index_symbol = node.slice.elts[0].id
            particle_index = self.kernel_globals[particle_index_symbol]
            cellx = particle_index.sym_loop_cell
            layerx = particle_index.sym_loop_layer

            component_index = node.slice.elts[1]

            new_node = ast.Subscript(
                value=ast.Subscript(
                    value=ast.Subscript(
                        value=ast.Name(id=symbol, ctx=ast.Load()),
                        slice=ast.Name(id=cellx, ctx=ast.Load()),
                        ctx=ast.Load(),
                    ),
                    slice=component_index,
                    ctx=ast.Load(),
                ),
                slice=ast.Name(id=layerx, ctx=ast.Load()),
                ctx=ast.Load(),
            )

        return ast.copy_location(new_node, node)


class Loop:
    def __init__(self, *args, **kwargs):

        self.kernel = args[0]
        self.args = args[1:]

        print(self.kernel)
        print(self.args)

        kernel_vars = inspect.getclosurevars(self.kernel)
        k_ast = ast.parse(inspect.getsource(self.kernel))

        kernel_params = k_ast.body[0].args.args

        kernel_args = {}
        for vi, varx in enumerate(kernel_params):
            kernel_args[varx.arg] = self.args[vi]

        particle_dat_rewrite = ParticleDatReWrite(kernel_args, kernel_vars.globals)

        k_ast = particle_dat_rewrite.visit(k_ast)

        print(ast.dump(k_ast, indent=2))

        visitor = GalleVisitor(self.args, kernel_vars.globals)
        visitor.visit(k_ast)

        for nx in visitor.body_nodes:
            print(nx)
