from CIMtools.preprocessing import Fragmentor
from multiprocessing import Pool
from CGRtools.files import SDFRead
import os
import pickle

with SDFRead('/home/alex/retrosynthesis/sdf/TestSetNew.sdf', 'r') as r:
    test = r.read()

os.environ["PATH"] = "/home/alex/retrosynthesis/frag"
fr = Fragmentor(remove_rare_ratio=0.00001)

if __name__ == '__main__':
    with Pool(6) as p:
        p.map_async(fr.partial_fit, test)

    fr.finalize()

    with open('fitted_fragmentor.pickle', 'wb') as f:
        pickle.dump(fr, f)
