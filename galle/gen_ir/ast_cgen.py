import ast
import cgen
from pymbolic.interop.ast import ASTToPymbolic
import pymbolic as pmbl

from galle.gen_ir.scope import *


class GalleVisitor(ast.NodeVisitor):
    def __init__(self, args, kernel_globals):
        ast.NodeVisitor.__init__(self)

        self.kernel_globals = kernel_globals
        self.args = args
        self.params_set = False
        self.kernel_locals = {}
        self.body_nodes = []
        self._ast2p = ASTToPymbolic()
        self.scope = Scope()

    def ast2p(self, node):
        return self._ast2p(node)

    def get_name(self, node):

        if issubclass(type(node), ast.Name):
            return node.id
        else:
            assert issubclass(type(node), ast.Subscript)
            return self.get_name(node.value)

    def visit_FunctionDef(self, node):
        self.scope.push()

        # Build the map from the kernel symbols to the concrete type for this
        # loop
        if self.params_set:
            raise RuntimeError("Cannot define functions in kernels")
        self.params_set = True

        for pi, px in enumerate(node.args.args):
            self.kernel_locals[px.arg] = self.args[pi]
        for kx in self.kernel_locals.keys():
            self.scope.add(kx)

        for nodex in node.body:
            self.scope.add_node(self.visit(nodex))

        self.body_nodes = self.scope.stack_nodes[0]

        self.scope.pop()

    def visit_Assign(self, node):

        assert len(node.targets) == 1

        def create_node(target, value):
            lvalue = self.ast2p(target)
            rvalue = self.ast2p(value)
            
            target_symbol = self.get_name(target)

            if self.scope.contains(target_symbol):
                return_node = cgen.Assign(lvalue, rvalue)
            else:
                self.scope.add(target_symbol)
                return_node = cgen.Statement(f"auto {lvalue} = {rvalue}")

            return return_node

        if not issubclass(type(node.targets[0]), ast.Tuple):
            target = node.targets[0]
            value = node.value
            return create_node(target, value)

        else:
            print(ast.dump(node, indent=2))

            nodes = []
            for targetx, valuex in zip(node.targets[0].elts, node.value.elts):
                nodes.append(create_node(targetx, valuex))

            for nx in nodes[:-1]:
                self.scope.add_node(nx)
            return nodes[-1]


    def visit_For(self, node):
        self.scope.push()

        if not type(node.iter) is ast.Call:
            raise RuntimeError("For loop must be python range.")
        if not node.iter.func.id == "range":
            raise RuntimeError("For loop must be python range.")

        loop_args = node.iter.args
        if len(loop_args) == 1:
            loop_start = 0
            loop_end = self.ast2p(loop_args[0])
            loop_inc = 1
        elif len(loop_args) == 2:
            loop_start = self.ast2p(loop_args[0])
            loop_end = self.ast2p(loop_args[1])
            loop_inc = 1
        elif len(loop_args) == 3:
            loop_start = self.ast2p(loop_args[0])
            loop_end = self.ast2p(loop_args[1])
            loop_inc = self.ast2p(loop_args[2])
        else:
            raise RuntimeError("unknown for loop type")

        loop_target = node.target.id

        for nx in node.body:
            self.scope.add_node(self.visit(nx))

        start = cgen.Line(f"int {loop_target} = {loop_start}")
        condition = cgen.Line(f"{loop_target} < {loop_end}")
        update = cgen.Line(f"{loop_target}+={loop_inc}")

        for_loop = cgen.For(start, condition, update, cgen.Block(self.scope.get_nodes()))

        self.scope.pop()
        return for_loop

    def visit_If(self, node):

        condition = self.ast2p(node.test)
        self.scope.push()
        for nodex in node.body:
            self.scope.add_node(self.visit(nodex))
        block = self.scope.get_nodes()
        self.scope.pop()
        self.scope.push()
        for nodex in node.orelse:
            self.scope.add_node(self.visit(nodex))
            block_else = self.scope.get_nodes()
        self.scope.pop()

        return cgen.If(condition, cgen.Block(block), cgen.Block(block_else))

    def visit_Num(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Str(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_FormattedValue(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_JoinedStr(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Bytes(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_List(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Tuple(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Set(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Dict(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Ellipsis(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_NameConstant(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Name(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Constant(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Load(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Store(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Del(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Starred(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Expr(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_UnaryOp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_UAdd(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_USub(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Not(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Invert(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Add(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Sub(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Mult(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Div(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_FloorDiv(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Mod(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Pow(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_LShift(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_RShift(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_BitOr(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_BitXor(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_BitAnd(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_BinOp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_MatMult(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_BoolOp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_And(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Or(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Compare(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Eq(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_NotEq(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Lt(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_LtE(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Gt(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_GtE(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Is(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_IsNot(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_In(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_NotIn(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Call(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_keyword(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_IfExp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Attribute(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Index(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Slice(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_ExtSlice(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_ListComp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_SetComp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_GeneratorExp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_DictComp(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_comprehension(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_AnnAssign(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_AugAssign(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Print(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Raise(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Assert(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Delete(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Pass(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Import(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_ImportFrom(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_alias(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_While(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Break(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Continue(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Try(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_TryFinally(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_TryExcept(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_ExceptHandler(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_With(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_withitem(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Lambda(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_arguments(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_arg(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Return(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Yield(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_YieldFrom(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Global(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Nonlocal(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_ClassDef(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_AsyncFunctionDef(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_Await(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_AsyncFor(self, node):
        raise RuntimeError("Bad construct in kernel.")

    def visit_AsyncWith(self, node):
        raise RuntimeError("Bad construct in kernel.")
