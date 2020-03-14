import networkx as nx
import torch
import torch.nn as nn
import random as rnd
import math as m
from CGRtools.files import SDFRead
from CGRtools.reactor import CGRReactor
from CIMtools.preprocessing import StandardizeChemAxon, Fragmentor
from sklearn.base import BaseEstimator

c_puct = 4

model = nn.Sequential(
    nn.Linear(2006, 4000),
    nn.ReLU(inplace=True),
    nn.Linear(4000, 2272),
    nn.Sigmoid()
)
model.load_state_dict(torch.load('model/model_dict.pth'))
model.eval()


class MCTS:
    def __init__(self, tree, model):
        self._tree = tree
        self._model = model
        self._fr = Fragmentor()

    def predict(self, mol_container):
        descriptor = self._fr.transform(mol_container).values
        return self._model(descriptor), 1

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
        reagent = self._tree.nodes[node]['reagents']
        rules, value = self.predict(reagent)
        for rule in rules:
            reactor = CGRReactor(rule)
            products = reactor(reagent)
            for product in products:
                self._tree.add_node(len(self._tree.nodes), reagents=product)
        return value

    def backup(self, node, value):
        parent = list(self._tree.predecessors(node))
        while parent:
            visit_count = self._tree.nodes[node]['visit_count'] + 1
            total_action = self._tree.nodes[node]['total_action'] + value
            mean_action = total_action / visit_count
            self._tree.add_node(node, visit_count=visit_count, total_action=total_action, mean_action=mean_action,
                                depth=nx.shortest_path_length(self._tree, 1, node))
            node = parent[0]
            parent = list(self._tree.predecessors(node))

    def train(self):
        ...

    def play(self):
        ...
