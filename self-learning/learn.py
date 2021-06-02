from CGRtools import smiles
from pytorch_lightning import Trainer
from ThetaSynthesis import RetroTree
from ThetaSynthesis.synthon import DoubleHeadedSynthon
from ThetaSynthesis.synthon.policy.model import RetroTreeDataModule, DoubleHeadedNet
from collector import Collector


class SmarterDoubleHeadedSynthon(DoubleHeadedSynthon):
    __paths__ = {'double': 'learning.ckpt'}


smileses = ['O=[N+]([O-])c2oc(/C=N/N1C(=O)NC(=O)C1)cc2']

molecules = []
for s in smileses:
    m = smiles(s)
    m.canonicalize()
    molecules.append(m)

net = DoubleHeadedNet()
c = Collector(bad=10)
c.fit()
forest = [RetroTree(m, DoubleHeadedSynthon, iterations=5e4) for m in molecules]

for _ in range(10):
    tensors = c.transform(forest)
    print(f'Tensor X shape is {tensors[0].shape}')
    print(f'Tensor y shape is {tensors[1].shape}')
    data = RetroTreeDataModule(tensors)

    trainer = Trainer(automatic_optimization=False, max_epochs=20)
    trainer.fit(net, data)
    trainer.save_checkpoint('learning.ckpt')

    forest = [RetroTree(m, SmarterDoubleHeadedSynthon, iterations=5e4) for m in molecules]
    for tree in forest:
        paths = list(tree)
        print(tree.report())
        print(paths)
        print(' '.join([str(tree._node_depth[x]) for x in paths]))

    forest = [RetroTree(m, SmarterDoubleHeadedSynthon, iterations=5e4) for m in molecules]
