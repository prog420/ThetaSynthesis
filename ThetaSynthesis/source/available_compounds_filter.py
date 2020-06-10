import pickle

with open('source files/new_bb.pickle', 'rb') as f:
    blocks = pickle.load(f)


def not_available(lst):
    return [mol for mol in lst if str(mol) not in blocks and len(mol) > 6]
