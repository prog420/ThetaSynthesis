from itertools import chain, repeat
from torch import hstack, vstack, tensor, zeros
from typing import Iterable, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ThetaSynthesis import RetroTree
    from torch import Tensor


class Collector:
    def __init__(self, good: int = 100, bad: int = 0):
        self._good = good
        self._bad = min(bad, good) if bad else bad

    def fit(self, examples: Optional[Tuple['Tensor', 'Tensor']] = None):
        self._X, self._y = examples if examples else tensor([]), tensor([])
        print(self._X)

    def transform(self, forest: Iterable['RetroTree']) -> Tuple['Tensor', 'Tensor']:
        win_examples, lose_examples = [], []
        for tree in forest:
            win_terminals = list(tree)
            winners = set(chain.from_iterable([tree.chain_to_node(node) for node in win_terminals]))

            losers = sorted([x for x in tree._nodes if not tree._succ[x] and x not in winners],
                            key=lambda k: tree._visits[k],
                            reverse=True)
            X = self._X
            for example in chain(zip(winners, repeat(1.)), zip(losers, repeat(-1.))):
                node, value = example

                vec_X, vec_y = tree._nodes[node].current_synthon._bit_string, tensor(value)
                if not X.size()[0] or vec_X not in X:
                    if value == 1. and len(win_examples) < self._good:
                        win_examples.append([vec_X, vec_y])
                    elif value == -1. and len(lose_examples) < self._bad:
                        lose_examples.append([vec_X, vec_y])

        examples = [*win_examples, *lose_examples]
        self._X = vstack([self._X, vstack([x[0] for x in examples])]) \
            if self._X.size()[0] \
            else vstack([x[0] for x in examples])
        self._y = vstack([self._y, vstack([x[1] for x in examples])]) \
            if self._y.size()[0] \
            else vstack([x[1] for x in examples])
        return self._X, self._y


__all__ = ['Collector']
