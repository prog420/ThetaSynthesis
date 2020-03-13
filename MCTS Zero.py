import networkx as nx
import torch
import random as rnd
import math as m
from CGRtools.files import SDFRead
from CGRtools.reactor import CGRReactor
from CIMtools.preprocessing import StandardizeChemAxon
from sklearn.base import BaseEstimator

c_puct = 4

"""
1 - Интересующее соединение, описывается фрагментными дескрипторами ISIDA. 
2 - Далее искусственной нейронной сетью предсказывается вероятность применения правил, по
которому можно получить целевое соединение. 
3 - КГР реакционного правила накладывается на соединение и генерируются реагенты, с помощью которых можно получить
интересующее соединение
"""


class MCTS:
    def __init__(self, tree):
        self._tree = tree

    def puct(self, node):
        global c_puct
        mean_action = self._tree.nodes[node]['mean_action']
        probability = self._tree.nodes[node]['probability']
        visit_count = self._tree.nodes[node]['visit_count']
        parent_node = list(tree.predecessors(node))[0]
        sum_visit_count = sum(
            [self._tree.nodes[node]['visit_count']
             for node
             in list(tree.successors(parent_node))
             ]
        )
        ucp = c_puct * probability * (m.sqrt(sum_visit_count) / (1 + visit_count))
        return mean_action + ucp

    def select(self):
        children = list(tree.successors(1))
        maximum = -10000
        while children:
            for child_node in children:
                if self.puct(child_node) > maximum:
                    node = child_node
                    maximum = self.puct(child_node)
            children = list(tree.successors(self.node))
        return node

    def expand_and_evaluate(node):
        reagent = tree.nodes[node]['reagents']
        rules = nn(reagent)
        for rule in rules:
            reactor = CGRReactor(rules)
            products = reactor(reagent)
            for product in products:
                tree.add_node(len(tree.nodes), reagents=product)
        return value

    def backup(node, value):
        parent = list(tree.predecessors(node))
        while parent:
            visit_count = tree.nodes[node]['visit_count'] + 1
            total_action = tree.nodes[node]['total_action'] + value
            mean_action = total_action / visit_count
            tree.add_node(node, visit_count=visit_count, total_action=total_action, mean_action=mean_action,
                          depth=nx.shortest_path_length(tree, 1, node))
            node = parent[0]
            parent = list(tree.predecessors(node))

    def play(self):
        ...


if __name__ == '__main__':
    with SDFRead('TargetSample.sdf', 'r') as f:
        target_list = f.read()
    model = torch.load('/home/alex/Desktop/retrosynthesis/sdf/full_model.pth')
    model.eval()
    target = rnd.choice(target_list)
    tree = nx.DiGraph()
    path = mcts(tree, target)
