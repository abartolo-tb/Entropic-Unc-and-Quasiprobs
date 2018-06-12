import numpy as np
import numpy.linalg as linalg
from simulationTools import *

nCores = 2
n = 8
hz = [.375, -.375, .375, -.375, .375, -.375, .375, -.375]
H = power_law_ising_hamiltonian(n, hz, 1.05, 1, 5, 6)
beta = 1
Lambda, Q = linalg.eigh(H.todense())
rho = thermal_density_matrix(Lambda, Q, beta)
vAxis = [0,0]
V1 = construct_local_projector(n, 0, 0, vAxis)
V2 = construct_local_projector(n, 0, 1, vAxis)
xValues = np.arange(-30,30,.1)
pg1 = weak_measurement_couplings(10, .1, .02, xValues)
pg2 = weak_measurement_couplings(10, .1, .02, xValues)
wAxis = [0,0]
W1 = construct_local_projector(n, n-1, 0, wAxis)
W2 = construct_local_projector(n, n-1, 1, wAxis)
Wlist = [W1, W2]
tList = np.linspace(0,10,1001)

LHSneu, LHSmm, LHSneuTerms, LHSmmTerms, rhoF, rhoR, f, fOrd, fTerm, qProbs = simulation(V1, V2, pg1, pg2, Wlist, H, rho, tList, nWorks=nCores)

np.save('./weak_data/tList.npy', tList)
np.save('./weak_data/LHSneu.npy', LHSneu)
np.save('./weak_data/LHSmm.npy', LHSmm)
np.save('./weak_data/LHSneuTerms.npy', LHSneuTerms)
np.save('./weak_data/LHSmmTerms.npy', LHSmmTerms)
np.save('./weak_data/rhoF.npy', rhoF)
np.save('./weak_data/rhoR.npy', rhoR)
np.save('./weak_data/f.npy', f)
np.save('./weak_data/fOrd.npy', fOrd)
np.save('./weak_data/fTerm.npy', fTerm)
np.save('./weak_data/qProbs.npy', qProbs)