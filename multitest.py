from CIMtools.preprocessing import Fragmentor
from CGRtools.files import SDFRead
import os
import pickle

with SDFRead('/home/alex/theta/TestSetNew.sdf', 'r') as r:
    test = r.read()

os.environ["PATH"] = "/home/alex/theta"
fr = Fragmentor(remove_rare_ratio=0.00001)

fr.partial_fit(test)

fr.finalize()

with open('fitted_fragmentor.pickle', 'wb') as f:
    pickle.dump(fr, f)
