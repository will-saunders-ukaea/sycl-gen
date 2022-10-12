import ast
import numbers

from galle.gen_particle.kernel_symbols import *


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


class ConstantReWrite(ast.NodeTransformer):
    def __init__(self, kernel_globals):
        ast.NodeTransformer.__init__(self)
        self.kernel_globals = kernel_globals

    def visit_Name(self, node):
        if node.id in self.kernel_globals:

            val = None
            obj = self.kernel_globals[node.id]
            if issubclass(type(obj), numbers.Number):
                val = obj
            elif issubclass(type(obj), Constant):
                val = obj.value

            if val is not None:
                new_node = ast.Constant(value=val)

                return ast.copy_location(new_node, node)
            else:
                return node

        return node
