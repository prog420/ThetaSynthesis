from CGRtools.files import SDFRead
from CGRtools.reactor import CGRReactor
from random import choice, sample
from sklearn.base import BaseEstimator
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


class Estimator(BaseEstimator):
    def __init__(self):
        ...

    def fit(self):
        ...

    def predict(self):
        ...


class MCTS:
    def __init__(self, target, stop: dict):
        self._target = target
        self._tree = nx.DiGraph()
        self._tree.add_node(1, depth=0, reactants=target, mean_action=0, visit_count=0, total_action=0,
                            probability=1)
        self._step_count, self._depth_count, self._node_child_count, self._terminal_count = stop.values()

    @staticmethod
    def predict(mol_container):
        descriptor = torch.FloatTensor(frag.transform([mol_container]).values)
        y = model(descriptor)
        list_rules = [x
                      for x in sorted(zip(range(1, len(y[0])), [i.item() for i in y[0]]), key=lambda x: x[1], reverse=True)
                      if x[1] >= 0.95
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
        return node

    def expand_and_evaluate(self, node):
        reactant = self._tree.nodes[node]['reactants']
        rules, value = self.predict(reactant)
        child_count = 0
        for pair in rules:
            if child_count >= self._node_child_count:
                break
            rule, probability = pair
            reactor = CGRReactor(rule)
            products = reactor(reactant)
            for product in products:
                new_node = len(self._tree.nodes) + 1
                self._tree.add_edge(node, new_node, rule=rule)
                self._tree.add_node(new_node, reactants=product, mean_action=0, visit_count=0, total_action=0,
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

    def train(self):
        ...

    def play(self):
        for _ in range(self._step_count):
            node = self.select()
            if nx.shortest_path_length(self._tree, 1, node) == self._depth_count:
                break
            value = self.expand_and_evaluate(node)
            self.backup(node, value)
        children = sorted(list(self._tree.successors(1)), key=lambda x: self._tree.nodes[x]['mean_action'], reverse=True)
        return children[0]


with SDFRead('./source files/TestSetNew.sdf', 'r') as file:
    test = file.read()
    targets = sample(test, 8)
    del test

target = choice(targets)
path = [target]
for _ in range(3):
    tree = MCTS(target, {'step_count': 100, 'depth_count': 10, 'node_child_count': 5, 'terminal_count': 10})
    target = tree.play()
    path.append(target)

pickle.dump(path, 'result.pickle')
