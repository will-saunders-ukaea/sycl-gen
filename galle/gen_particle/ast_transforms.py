import ast
import inspect
import numbers
import copy
from typing import Callable

import pymbolic as pmbl

from galle.gen_particle.kernel_symbols import *
from galle.gen_ir.scope import *


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

    deps = {"node": func, "node_ast": func_ast, "node_globals": func_globals, "deps": {}}
    for gx in func_globals.items():
        deps["deps"][gx[0]] = get_dependencies(gx[1])

    return deps


def function_rewrite_constants(deps):

    for depx in deps["deps"].items():
        function_rewrite_constants(depx[1])

    func = deps["node"]
    if is_rewriteable_func(func):
        func_globals = deps["node_globals"]
        func_ast = deps["node_ast"]
        constant_rewrite = ConstantReWrite(func_globals)
        func_new_ast = constant_rewrite.visit(func_ast)
        deps["node_ast"] = func_new_ast


def get_exising_names(kernel_ast):
    names = set()
    for node in ast.walk(kernel_ast):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def extract_function_def(nodes):
    for node in ast.walk(nodes):
        if isinstance(node, ast.FunctionDef):
            return node


class NameRenamer(ast.NodeTransformer):
    def __init__(self, call_args, names, output_scope):
        ast.NodeTransformer.__init__(self)
        self.call_args = call_args
        self.name_generator = UniqueNamesGenerator(names)
        self.name_map = {}
        self.output_scope = output_scope
        self.return_name = None

    def visit_FunctionDef(self, node):
        if len(node.args.posonlyargs) != 0:
            raise NotImplementedError("Position only args not implemented.")
        if len(node.args.kwonlyargs) != 0:
            raise NotImplementedError("Keyword args not implemented.")
        if len(node.args.defaults) != 0:
            raise NotImplementedError("Default args not implemented.")

        for parami, paramx in enumerate(node.args.args):
            internal_name = self.name_generator(paramx.arg)
            self.name_map[paramx.arg] = internal_name
            self.output_scope.add_node(
                ast.copy_location(
                    ast.Assign(targets=[ast.Name(id=internal_name, ctx=ast.Store())], value=self.call_args[parami]),
                    node,
                )
            )

        for bx in node.body:
            n = self.visit(bx)
            if n is not None:
                self.output_scope.add_node(n)

    def visit_Name(self, node):

        if node.id in self.name_map.keys():
            new_node = copy.deepcopy(node)
            new_node.id = self.name_map[node.id]
        else:
            new_node = copy.deepcopy(node)
            new_id = self.name_generator(node.id)
            new_node.id = new_id
            self.name_map[node.id] = new_id

        return ast.copy_location(new_node, node)

    def visit_Return(self, node):
        self.return_name = self.visit(node.value)
        return None


class FunctionInline(ast.NodeTransformer):
    def __init__(self, func_deps):
        ast.NodeTransformer.__init__(self)
        self.func_deps = func_deps
        self.scope = Scope()

        self.name_generator = None

    def visit(self, node):
        self.name_generator = UniqueNamesGenerator(get_exising_names(node))
        return super().visit(node)

    def generic_visit_child_nodes(self, node):
        self.scope.push()

        for nodex in node.body:
            self.scope.add_node(self.visit(nodex))
        new_node = copy.deepcopy(node)
        new_node.body = self.scope.get_nodes()

        self.scope.pop()
        return ast.copy_location(new_node, node)

    def visit_FunctionDef(self, node):
        return self.generic_visit_child_nodes(node)

    def visit_If(self, node):
        return self.generic_visit_child_nodes(node)

    def visit_For(self, node):
        return self.generic_visit_child_nodes(node)

    def visit_Call(self, node):

        # find the corresponding ast for the call
        function_name = node.func.id

        if function_name in self.func_deps.keys():
            function_ast = extract_function_def(self.func_deps[function_name]["node_ast"])

            call_args = node.args

            name_renamer = NameRenamer(call_args, self.name_generator.names, self.scope)
            name_renamer.visit(function_ast)
            self.name_generator.add(name_renamer.name_generator.names)

            return_name = name_renamer.return_name
            if return_name is None:
                raise RuntimeError("Inlined function did not return a value.")
            new_node = ast.copy_location(return_name, node)

            return new_node
        else:
            return node


def inline_functions(deps):
    for depx in deps["deps"].items():
        inline_functions(depx[1])

    func = deps["node"]
    if is_rewriteable_func(func):
        function_inline = FunctionInline(deps["deps"])
        new_ast = function_inline.visit(deps["node_ast"])
        deps["node_ast"] = new_ast
