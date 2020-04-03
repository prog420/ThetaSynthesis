from available_compounds_filter import not_available
from CGRtools.containers import ReactionContainer
from CGRtools.files import SDFRead
from CGRtools.reactor import CGRReactor
from random import choice
import math as m
import networkx as nx
import pickle
import torch

c_puct = 4
with open('./source files/fitted_fragmentor.pickle', 'rb') as f:
    frag = pickle.load(f)
with open('./source files/rules_reverse.pickle', 'rb') as f:
    rules = pickle.load(f)

model = torch.load('./source files/full_model.pth')
model.eval()


class MCTS:
    def __init__(self, target, stop):
        self._target = target
        self._tree = nx.DiGraph()
        self._tree.add_node(1, depth=0, queue=[target], mean_action=0, visit_count=0, total_action=0,
                            probability=1.)
        self._terminal_nodes = []
        self._stop = stop

    @property
    def terminal_nodes(self):
        return self._terminal_nodes

    @staticmethod
    def nn(mol_container):
        descriptor = torch.FloatTensor(frag.transform([mol_container]).values)
        y = model(descriptor)
        list_rules = [x
                      for x in sorted(zip(range(1, len(y[0])), [i.item() for i in y[0]]), key=lambda x: x[1], reverse=True)
                      if x[1] >= 0.995
                      ]
        list_rules = [(rules[x], y) for x, y in list_rules]
        return list_rules, 1

    @staticmethod
    def filter(reaction):
        return True

    def puct(self, node):
        global c_puct
        mean_action = self._tree.nodes[node]['mean_action']
        probability = self._tree.nodes[node]['probability']
        visit_count = self._tree.nodes[node]['visit_count']
        parent_node = list(self._tree.predecessors(node))[0]
        sum_visit_count = sum(
            [self._tree.nodes[node]['visit_count']
             for node
             in list(self._tree.successors(parent_node))
             ]
        )
        ucp = c_puct * probability * (m.sqrt(sum_visit_count) / (1 + visit_count))
        return mean_action + ucp

    def select(self):
        node = 1
        children = list(self._tree.successors(node))
        maximum = -10000
        while children:
            for child_node in children:
                if self.puct(child_node) > maximum:
                    node = child_node
                    maximum = self.puct(child_node)
            children = list(self._tree.successors(node))
            maximum = -10000
        return node

    def expand_and_evaluate(self, node):
        reactant = self._tree.nodes[node]['queue'].pop(0)
        rules, value = self.nn(reactant)
        for pair in rules:
            rule, probability = pair
            probability /= len(pair)
            reactor = CGRReactor(rule)
            products = list(reactor(reactant))
            if products:
                self._tree.add_edge(node, len(self._tree.nodes) + 1, rule=rule,
                                    reaction=ReactionContainer([reactant], products))
                comm_products = not_available(products)
                queue = self._tree.nodes[node]['queue'] + comm_products
                self._tree.add_node(len(self._tree.nodes), queue=queue, mean_action=0, visit_count=0, total_action=0,
                                    depth=nx.shortest_path_length(self._tree, 1, node),
                                    probability=probability)
        return value

    def backup(self, node, value):
        parent = list(self._tree.predecessors(node))
        while parent:
            visit_count = self._tree.nodes[node]['visit_count'] + 1
            total_action = self._tree.nodes[node]['total_action'] + value
            mean_action = total_action / visit_count
            self._tree.add_node(node, visit_count=visit_count, total_action=total_action, mean_action=mean_action)
            node = parent[0]
            parent = list(self._tree.predecessors(node))

    def emulate(self):
        step_count, depth_count, terminal_count = self._stop.values()
        for _ in range(step_count):
            node = self.select()
            if nx.shortest_path_length(self._tree, 1, node) > depth_count:
                break
            self.backup(node, self.expand_and_evaluate(node))
            self._terminal_nodes = [node
                                    for node
                                    in self._tree.nodes
                                    if (not self._tree.nodes[node]['queue']) and (node != 1)
                                    ]
            if len(self._terminal_nodes) >= terminal_count:
                break

    def train(self, win_lose: dict = None):
        ...

    def find(self):
        self.emulate()
        if not self._terminal_nodes:
            return None
        paths = [nx.shortest_path(self._tree, 1, node) for node in self._terminal_nodes]
        for path in paths:
            lst = list(zip(path, path[1:]))
            reactions = [self._tree.edges[edge]['reaction'] for edge in lst]
            yield reactions


def main():
    with SDFRead('./source files/targets.sdf', 'r') as file:
        targets = file.read()
    target = choice(targets)
    # target = target[0]
    if not not_available([target]):
        print('Target can be bought')
        return
    tree = MCTS(target, {'step_count': 100, 'depth_count': 10, 'terminal_count': 10})
    paths = list(tree.find())
    with open('result.pickle', 'wb') as f:
        pickle.dump(paths, f)
    print('All done')


if __name__ == '__main__':
    main()
