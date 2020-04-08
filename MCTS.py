from available_compounds_filter import not_available
from CGRtools.containers import ReactionContainer
from CGRtools.files import SDFRead
from CGRtools.reactor import CGRReactor
from math import sqrt
from Model import Chem
from random import choice
from time import time
import networkx as nx
import pickle
import torch

c_puct = 4
flag_value = False
with open('./source files/fitted_fragmentor.pickle', 'rb') as f:
    frag = pickle.load(f)
with open('./source files/rules_reverse.pickle', 'rb') as f:
    rules = pickle.load(f)

policy_value = Chem(2006, 2273)
policy_value.load_state_dict(torch.load('./source files/twohead_state_dict.pth'))
policy_value.eval()

only_policy = torch.load('./source files/full_model.pth')
only_policy.eval()


class MCTS:
    def __init__(self, target, stop):
        self._target = target
        self._tree = nx.DiGraph()
        self._tree.add_node(1, depth=0, queue=[target], mean_action=0, visit_count=0, total_action=0,
                            probability=1.)
        self._terminal_nodes = []
        self._step_count, self._depth_count, self._terminal_count = stop['step_count'], \
                                                                    stop['depth_count'], \
                                                                    stop['terminal_count']

    @property
    def terminal_nodes(self):
        return self._terminal_nodes

    @staticmethod
    def filter(reaction):
        return True

    def puct(self, node):
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
        ucp = c_puct * probability * (sqrt(sum_visit_count) / (1 + visit_count))
        return mean_action + ucp

    def select(self):
        node = 1
        children = list(self._tree.successors(node))
        while children:
            node = max(children, key=self.puct)
            children = list(self._tree.successors(node))
        return node

    @staticmethod
    def nn(mol_container):
        descriptor = torch.FloatTensor(frag.transform([mol_container]).values)
        if flag_value:
            y = policy_value(descriptor)
            list_rules = [x
                          for x in
                          sorted(enumerate([i.item() for i in y[0][0]], start=1), key=lambda x: x[1], reverse=True)
                          if x[1] >= 0.995
                          ]
            list_rules = [(rules[x], y) for x, y in list_rules]
            return list_rules, y[1][0][0].item()
        else:
            y = only_policy(descriptor)
            list_rules = [x
                          for x in
                          sorted(enumerate([i.item() for i in y[0]], start=1), key=lambda x: x[1], reverse=True)
                          if x[1] >= 0.995
                          ]
            list_rules = [(rules[x], y) for x, y in list_rules]
            return list_rules

    def expand_and_evaluate(self, node):
        reactant = self._tree.nodes[node]['queue'].pop(0)
        if flag_value:
            rules, value = self.nn(reactant)
        else:
            rules, value = self.nn(reactant), self.rollout(node, reactant)
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

    def rollout(self, node, mol_container):
        len_rollout = self._depth_count - nx.shortest_path_length(self._tree, 1, node)
        queue = [mol_container]
        for _ in range(len_rollout):
            reactant = queue.pop(0)
            descriptor = torch.FloatTensor(frag.transform([reactant]).values)
            y = only_policy(descriptor)
            list_rules = [x for x in
                          sorted(enumerate([i.item() for i in y[0]], start=1), key=lambda x: x[1], reverse=True)
                          if x[1] == 1
                          ]
            list_rules = [(rules[x], y) for x, y in list_rules]
            for rule in list_rules:
                reactor = CGRReactor(rule[0])
                products = list(reactor(reactant))
                if products:
                    queue.extend(not_available(products))
                    break
            if not queue:
                return 1
        return 0

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
        for _ in range(self._step_count):
            node = self.select()
            if nx.shortest_path_length(self._tree, 1, node) > self._depth_count:
                break
            self.backup(node, self.expand_and_evaluate(node))
            self._terminal_nodes = [node
                                    for node
                                    in self._tree.nodes
                                    if (not self._tree.nodes[node]['queue']) and (node != 1)
                                    ]
            if len(self._terminal_nodes) >= self._terminal_count:
                break

    def train(self, win_lose: dict = None):
        ...

    def find(self):
        self.emulate()
        if not self._terminal_nodes:
            return None
        paths = [nx.shortest_path(self._tree, 1, node)
                 for node
                 in self._terminal_nodes]
        paths = [x for x in paths if len(x) == len(max(paths, key=len))]
        for path in paths:
            lst = list(zip(path, path[1:]))
            reactions = [self._tree.edges[edge]['reaction'] for edge in lst]
            # if not not_available(reactions[-1].products):
            yield reactions


def main():
    start_time = time()
    with SDFRead('./source files/targets.sdf', 'r') as file:
        targets = file.read()
    target = choice(targets)
    # target = target[0]
    if not not_available([target]):
        print('Target can be bought')
        return
    tree = MCTS(target, {'step_count': 1000, 'depth_count': 10, 'terminal_count': 5})
    paths = list(tree.find())
    with open('result.pickle', 'wb') as f:
        pickle.dump(paths, f)
    print(f'All done, time = {time() - start_time} seconds')


if __name__ == '__main__':
    main()
