import pickle

with open('source files/blocks.pickle', 'rb') as f:
    blocks = pickle.load(f)


def available(lst):
    return [mol for mol in lst if str(mol) in blocks]
