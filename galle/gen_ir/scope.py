class Scope:
    def __init__(self):
        self.stack = []
        self.stack_nodes = []

    def add(self, key):
        self.stack[-1].append(key)

    def add_node(self, node):
        self.stack_nodes[-1].append(node)

    def get_nodes(self):
        return self.stack_nodes[-1]

    def push(self):
        self.stack.append([])
        self.stack_nodes.append([])

    def pop(self):
        self.stack.pop()
        self.stack_nodes.pop()

    def contains(self, key):
        for levelx in self.stack:
            if key in levelx:
                return True
        return False


class UniqueNamesGenerator:
    def __init__(self, names=set()):
        self.names = names
        self.counter = 0

    def __call__(self, basename="l"):
        def next_name():
            proposed_name = f"{basename}_{self.counter}"
            self.counter += 1
            return proposed_name

        pname = next_name()
        while pname in self.names:
            pname = next_name()

        self.names.add(pname)
        return pname

    def add(self, names):
        self.names.union(names)
