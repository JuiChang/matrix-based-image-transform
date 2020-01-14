import numpy as np
import pickle

dctVar = pickle.load(open("dct.pkl", "rb"))
whtVar = pickle.load(open("wht.pkl", "rb"))
dftVar = pickle.load(open("dft.pkl", "rb"))

print((dctVar - whtVar).sum())
print((dctVar - whtVar))

print((dctVar - dftVar).sum())
print((dctVar - dftVar))

print((whtVar - dftVar).sum())
print((whtVar - dftVar))

print("start using abs")

dctVar = np.absolute(dctVar)
whtVar = np.absolute(whtVar)
dftVar = np.absolute(dftVar)

print(dctVar.sum() - whtVar.sum())
print(dctVar.sum() - dftVar.sum())
print(whtVar.sum() - dftVar.sum())