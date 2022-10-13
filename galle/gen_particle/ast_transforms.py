import ast
import inspect
import numbers
from typing import Callable

import pymbolic as pmbl

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

        return node


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


def is_rewriteable_func(func):
    return issubclass(type(func), Callable) and not issubclass(type(func), pmbl.primitives.Variable)


def get_dependencies(func):

    func_globals = {}
    func_ast = None
    if is_rewriteable_func(func):
        func_globals = inspect.getclosurevars(func).globals
        func_ast = ast.parse(inspect.getsource(func))

    deps = {
            "node": func,
            "node_ast": func_ast,
            "node_globals": func_globals,
            "deps": {}
    }
    for gx in func_globals.items():
        deps["deps"][gx[0]] = get_dependencies(gx[1])

    return deps


def function_rewrite_constants(deps):
    
    for depx in deps["deps"].items():
        function_rewrite_constants(depx[1])
    
    func = deps["node"]
    if is_rewriteable_func(func):
        func_globals = deps["node_globals"]
        func_ast= deps["node_ast"]
        constant_rewrite = ConstantReWrite(func_globals)
        func_new_ast = constant_rewrite.visit(func_ast)
        deps["node_ast"] = func_new_ast


def get_exising_names(kernel_ast):

    names = set()
    for node in kernel_ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)

    return names


class FunctionInline(ast.NodeTransformer):
    def __init__(self, funcs):
        ast.NodeTransformer.__init__(self)
        self.funcs = funcs

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


def inline_functions(func_ast, func_globals):
    pass
















