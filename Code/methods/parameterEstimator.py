import numpy as np
import math
import scipy.linalg as la

D1 = np.array([[1,4],[1,4],[1,2],[1,4],[3,3]]).T
D2 = np.array([[1,5],[5,5],[1,6],[1,8],[3,7]]).T



## WRITE D1 and D2, get cov, mean, pw etc


print("Parameter estimation")
pw1 =len(D1[0])/(len(D1[0])+len(D2[0]))

pw2 =len(D2[0])/(len(D1[0])+len(D2[0]))
print("PW1", len(D1[0])/(len(D1[0])+len(D2[0])))
print("PW2", len(D2[0])/(len(D1[0])+len(D2[0])))
print("Mean D1:")
print(np.mean(D1, axis=1))
print("SigmaD1: ")
print(np.cov(D1))
print("Mean D2:")
print( np.mean(D2, axis=1))
print("SigmaD2: ")
print( np.cov(D2))
