import numpy as np
import numpy.linalg as linalg
from simulationTools import *

nCores = 2
n = 8
hz = [.375, -.375, .375, -.375, .375, -.375, .375, -.375]
H = power_law_ising_hamiltonian(n, hz, 1.05, 1, 5, 6)

vAxis = [0,0]
V1 = construct_local_projector(n, 0, 0, vAxis)
V2 = construct_local_projector(n, 0, 1, vAxis)
xValues = np.arange(-30,30,.1)
pg1 = weak_measurement_couplings(10, .1, .16, xValues)
pg2 = weak_measurement_couplings(10, .1, .16, xValues)
wAxisList = [[0,0], ]*n
Wlist = construct_complete_projector_list(n, wAxisList)
tList = np.linspace(0,10,101)

t_unscramble = 5
Lambda, Q = linalg.eigh(H.todense())
rho_unscrambled = Wlist[0]
U_scramble = get_U(Lambda, Q, t_unscramble)
rho = U_scramble.getH() @ rho_unscrambled @ U_scramble

LHSneu, LHSmm, LHSneuTerms, LHSmmTerms, rhoF, rhoR, f, aOrd, aTerm, qProbs = simulation_strong(V1, V2, pg1, pg2, Wlist, H, rho, tList, nWorks=nCores)

np.save('./strong_data/tList.npy', tList)
np.save('./strong_data/LHSneu.npy', LHSneu)
np.save('./strong_data/LHSmm.npy', LHSmm)
np.save('./strong_data/LHSneuTerms.npy', LHSneuTerms)
np.save('./strong_data/LHSmmTerms.npy', LHSmmTerms)
np.save('./strong_data/rhoF.npy', rhoF)
np.save('./strong_data/rhoR.npy', rhoR)
np.save('./strong_data/f.npy', f)
np.save('./strong_data/aOrd.npy', aOrd)
np.save('./strong_data/aTerm.npy', aTerm)
np.save('./strong_data/qProbs.npy', qProbs)