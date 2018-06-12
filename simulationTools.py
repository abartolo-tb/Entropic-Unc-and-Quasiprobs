import numpy as np
import numpy.linalg as linalg
import scipy as sp
import scipy.sparse as sparse
import scipy.special as special
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp


def bitvector_to_index(v):
    '''
    Given a list or array of binary values (v), convert to 
    an integer. The list is assumed to be big-endian.
    
    This method is meant to convert back and forth between
    matrix indices and spin configurations. In this sense,
    zeros are interpretted as spins in the +z state while
    ones are spins in the -z state.

    Arguments:
        v:      One dimensional list or array of binary 
                (0 or 1) values.
    
    Returns:
        ind:    Integer corresponding to big-endian
                interpretation of the binary list v.
    '''
    ind = sum([vj*(2**(len(v)-j-1)) for j,vj in enumerate(v)])
    return ind


def index_to_bitvector(ind, n):
    '''
    Given a non-negative interger (ind) and a number of 
    bits (n), return a big-endian binary list of length n 
    encoding ind. Require 0 <= ind < 2**n.
    
    This method is meant to convert back and forth between
    matrix indices and spin configurations. In this sense,
    zeros are interpretted as spins in the +z state while
    ones are spins in the -z state. n is the total number
    of spins.

    Arguments:
        ind:    Integer to be converted. Must be in 
                range(2**n).
        
        n:      Integer representing total number of 
                bits / spin sites.
    
    Returns:
        v:      Length n list of binary (0 or 1) values
                to be interpretted as big-endian.
    '''
    assert(isinstance(ind, int))
    assert(isinstance(n, int))
    assert(n >= 1)
    assert(ind in range(2**n))
    v = [int(bit) for bit in format(ind, '0'+str(n)+'b')]
    return v


def construct_local_pauli(nSpins, nMeas, axis):
    '''
    For a system with nSpins total spins, constructs the
    operator corresponding to the Pauli operator along 
    the specified axis at the specified spin (nMeas).
    
    Arguments:
        nSpins: Integer representing total number of spin 
                sites.
        
        nMeas:  Integer representing the spin site where
                the operator will act. Indexed from zero,
                thus must be in range(nSpins).
        
        axis:   A list of the form [theta, phi] specifying
                the axis of the Pauli operator's action.
                Theta is the polar angle given by a float
                in [0,pi] while phi is the azimuthal angle 
                given by a float in [0,2*pi).
    
    Returns:
        P:      A two-dimensional sparse CSR-matrix of size 
                (2**nSpins x 2**nSpins) representing the 
                action of the local Pauli operator on the 
                full Hilbert space.
    '''
    assert(isinstance(nSpins, int))
    assert(nSpins >= 1)
    assert(isinstance(nMeas, int))
    assert(nMeas in range(nSpins))
    assert(len(axis)==2)
    theta, phi = axis
    # Define 2x2 pauli matrix for the chosen axis
    sigma = np.array( [ [np.cos(theta), \
                         np.sin(theta)*np.exp(-1j*phi)], \
                        [np.sin(theta)*np.exp(1j*phi),\
                         -np.cos(theta)] ], dtype='complex')
    # Define operator on full Hilbert space as a sparse
    # COO-matrix
    row = []
    col = []
    data = []
    for ind in range(2**nSpins):
        # Convert current index to a binary list. Find
        # index for equivalent state with just the local
        # spin nMeas flipped.
        v = index_to_bitvector(ind,nSpins)
        vFlip = v.copy()
        vFlip[nMeas] = 1 - vFlip[nMeas]
        indFlip = bitvector_to_index(vFlip)
        # Using the 2x2 Pauli matrix sigma, add the
        # contributions from these two states to the
        # Pauli matrix on the full Hilbert space.
        s_site = v[nMeas]
        sFlip_site = vFlip[nMeas]
        row.append(ind)
        col.append(ind)
        data.append(sigma[s_site,s_site])
        row.append(ind)
        col.append(indFlip)
        data.append(sigma[s_site,sFlip_site])
    P = sparse.coo_matrix((data, (row,col)), \
                          shape=(2**nSpins,2**nSpins), \
                          dtype='complex')
    # Convert to sparse CSR-matrix
    P = P.tocsr()
    return P


def construct_local_projector(nSpins, nMeas, m, axis):
    '''
    For a system with nSpins total spins, constructs the
    operator corresponding to the projection operator along 
    the specified axis at the specified spin (nMeas) onto
    the subspace given by the value of m.
    
    Arguments:
        nSpins: Integer representing total number of spin 
                sites.
        
        nMeas:  Integer representing the spin site where
                the operator will act. Indexed from zero,
                thus must be in range(nSpins).
        
        m:      The integer 1 or 0. Corresponds to either 
                projecting on to the subspace of spin 
                aligned with the axis (0) or anti-aligned
                with the axis (1).
        
        axis:   A list of the form [theta, phi] specifying
                the axis of the projection operator's action.
                Theta is the polar angle given by a float
                in [0,pi] while phi is the azimuthal angle 
                given by a float in [0,2*pi).
    
    Returns:
        P:      A two-dimensional sparse CSR-matrix of size
                (2**nSpins x 2**nSpins) representing the 
                action of the local projection operator on 
                the full Hilbert space.
    '''
    assert(isinstance(nSpins, int))
    assert(nSpins >= 1)
    assert(isinstance(nMeas, int))
    assert(nMeas in range(nSpins))
    assert(m==0 or m==1)
    assert(len(axis)==2)
    # Set sign of projector.
    s = 1 - 2*m
    I = sparse.identity(2**nSpins,format='csr')
    pauli = construct_local_pauli(nSpins,nMeas,axis)
    P = .5*(I+s*pauli)
    return P


def construct_global_projector(nSpins, pState, axisL):
    '''
    For a system with nSpins total spins, constructs the
    operator corresponding to the projection operator along 
    the specified axis at the specified spin (nMeas) onto
    the subspace given by the sign of s.
    
    Arguments:
        nSpins: Integer representing total number of spin 
                sites.
        
        pState: A list of length nSpins consisting of the
                integers 0 and 1 or None. Each integer denotes 
                the orientation of the spin along the
                corresponding measurement axis. Zeros are 
                interpretted as spins aligned along the axis
                while ones are anti-aligned. None indicates no
                measurement is performed and must be matched
                with a corresponding None axis. The returned
                projection operator projects onto this state.
        
        axisL:  A list of length nSpins consisting of axes 
                where each axis is specified by the form 
                [theta, phi] or None. Theta is the polar angle 
                given by a float in [0,pi] while phi is the 
                azimuthal angle given by a float in  [0,2*pi). 
                Each axis specifies the orientation of the 
                projection operator's action on the 
                corresponding spin. If no measurement is to
                be performed on the spin, the corresponding 
                axis can be set to None. This must be matched
                with a corresponding None in pState.
    
    Returns:
        P:      A two-dimensional numpy-matrix of size
                (2**nSpins x 2**nSpins) representing the 
                projection operator onto the state specified
                by pState and axisL.
    '''
    assert(isinstance(nSpins, int))
    assert(nSpins >= 1)
    assert(len(pState)==nSpins)
    assert(len(axisL)==nSpins)
    assert(all([axis==None or len(axis)==2 for axis in axisL]))
    assert(all([m==0 or m==1 or m==None for m in pState]))
    P = 1
    for m, axis in zip(pState, axisL):
        try:
            # Axis and projection specified
            s = 1-2*m
            theta, phi = axis
            sigma = np.array( [ [np.cos(theta), \
                                 np.sin(theta)*np.exp(-1j*phi)], \
                                [np.sin(theta)*np.exp(1j*phi),\
                                 -np.cos(theta)] ], dtype='complex')
            I = np.identity(2)
            t = .5*(np.identity(2)+s*sigma)
        except TypeError:
            # No projection specified
            t = np.identity(2)
        P = np.kron(P, t)
    return P


def construct_complete_projector_list(nSpins, axisL):
    '''
    For a system with nSpins total spins, constructs all
    possible projection operators corresponding to possible
    measurement outcomes along the list of axes provided 
    (axisL). The list of returned projection operators is 
    complete.
    
    Arguments:
        nSpins: Integer representing total number of spin 
                sites.
        
        axisL:  A list of length nSpins consisting of axes 
                where each axis is specified by the form 
                [theta, phi] or None. Theta is the polar angle 
                given by a float in [0,pi] while phi is the 
                azimuthal angle given by a float in  [0,2*pi). 
                Each axis specifies the orientation of the 
                projection operator's action on the 
                corresponding spin. If no measurement is to
                be performed on the spin, the corresponding 
                axis can be set to None.
    
    Returns:
        Plist:  A list of two-dimensional numpy-matrices each
                of size (2**nSpins x 2**nSpins) representing 
                a projection operator onto a fine-grained 
                state.
    '''
    assert(isinstance(nSpins, int))
    assert(nSpins >= 1)
    assert(len(axisL)==nSpins)
    assert(all([axis==None or len(axis)==2 for axis in axisL]))
    # List possible projective measurements for each spin
    mList = [ [None] if axis==None else [0,1] for axis in axisL]
    # Generate all possible combinations of projective 
    # measurements
    stateList = list(itertools.product(*mList))
    # Generate complete list of projectors
    Plist = []
    for pState in stateList:
        P = construct_global_projector(nSpins, pState, axisL)
        Plist.append(P)
    return Plist


def transverse_field_ising_hamiltonian(nSpins, h, g, J):
    '''
    For a system with nSpins total spins, constructs the
    Hamiltonian for the transverse-field Ising model. The
    strength of nearest-neighbor interactions is set by J,
    while h is the transverse field along the z-axis, and 
    g is the longitudinal field along the x-axis.
    
    Hamiltonian can be written as,
    H = - J sum_{i} \sigma^{z}_{i} \sigma^{z}_{i+1}
        - h sum_{i} \sigma^{z}_{i}
        - g sum_{i} \sigma^{x}_{i}
    
    Arguments:
        nSpins: Integer representing total number of spin 
                sites.
        
        h:      Float giving the strength of the external
                field in the z-axis.
        
        g:      Float giving the strength of the external
                field in the x-axis.
        
        J:      Float giving the strength of nearest-
                neighbor interactions.
    
    Returns:
        H:      A two-dimensional sparse CSR-matrix of size 
                (2**nSpins x 2**nSpins) representing the 
                Hamiltonian of the system.
    '''
    assert(isinstance(nSpins, int))
    assert(nSpins >= 1)
    # Define the Hamiltonian as a COO-sparse matrix
    row = []
    col = []
    data = []
    for ind in range(2**nSpins):
        v = index_to_bitvector(ind,nSpins)
        # For given spin configuration, calculate the 
        # contributions from the transverse field and 
        # spin-spin interactions
        nUp = sum([s==0 for s in v])
        c = sum([(-1)**(v[i]+v[i+1]) for i in range(nSpins-1)])
        row.append(ind)
        col.append(ind)
        data.append(-h*(2*nUp-nSpins)-J*c)
        for n in range(nSpins):
            # Flip a single spin from the current
            # spin configuration
            vFlip = v.copy()
            vFlip[n] = 1 - vFlip[n]
            indFlip = bitvector_to_index(vFlip)
            # Calculate the coupling from the
            # longitudinal field.
            row.append(ind)
            col.append(indFlip)
            data.append(-g)
    H = sparse.coo_matrix((data, (row,col)), \
                          shape=(2**nSpins,2**nSpins), \
                          dtype='float')
    # Convert to sparse CSR-matrix
    H = H.tocsr()
    return H


def power_law_ising_hamiltonian(nSpins, h, g, J, lmax, zeta):
    '''
    For a system with nSpins total spins, constructs the
    Hamiltonian for the power-law Ising model. hList 
    specifies the transverse field along the z-axis at each
    site, and gList is the longitudinal field along the 
    x-axis at each site. If either of these are given as
    a scalar instead of a list, the field is assumed to be 
    constant. The strength of the spin-spin interaction is 
    set by J. lmax sets the maximum distance of the 
    interaction (measured in spin sites) while zeta 
    specifies the power at which the interaction decays.
    
    Hamiltonian can be written as,
    H = - sum_{l=1}^{lmax} sum_{i=1}^{nSpins-l} 
                (J/l**zeta) * \sigma^{z}_{i} \sigma^{z}_{i+l}
        - sum_{i} h_{i} \sigma^{z}_{i}
        - sum_{i} g_{i} \sigma^{x}_{i}
    
    Arguments:
        nSpins: Integer representing total number of spin 
                sites.
        
        h:      List of length nSpins consisting of floats 
                giving the strength of the external field in 
                the z-axis at each spin site. Can also be
                provided as a single float if the field is
                uniform.
        
        g:      List of length nSpins consisting of floats 
                giving the strength of the external field in 
                the x-axis at each spin site. Can also be
                provided as a single float if the field is
                uniform.
        
        J:      Float giving the strength of spin-spin
                interactions.
    
        lmax:   Maximum distance of the spin-spin interaction
                measured in number of lattice sites. Must be
                an integer between 0 (no interaction) and 
                nSpins-1 (all pair-wise interactions).
    
        zeta:   Non-negative float describing the power at
                which spin-spin interactions decay with 
                distance. Scaling is of the form 1 / l**zeta.
    
    Returns:
        H:      A two-dimensional sparse CSR-matrix of size 
                (2**nSpins x 2**nSpins) representing the 
                Hamiltonian of the system.
    '''
    assert(isinstance(nSpins, int))
    assert(nSpins >= 1)
    assert(isinstance(lmax, int))
    assert(0<=lmax and lmax<=nSpins-1)
    assert(zeta >= 0)
    # Define the Hamiltonian as a COO-sparse matrix
    row = []
    col = []
    data = []
    for ind in range(2**nSpins):
        v = index_to_bitvector(ind,nSpins)
        # Contribution from spin-spin interactions 
        Hss = J*sum([ sum([(-1)**(v[i]+v[i+l])/(l**zeta) for i in range(nSpins-l)]) 
                     for l in range(1,lmax+1)])
        try:
            # Calculate transverse field contribution
            Hz = sum([h[i]*(1-2*s) for i,s in enumerate(v)])
        except TypeError:
            # Case where transverse field is uniform
            Hz = h*sum([(1-2*s) for s in v])
        row.append(ind)
        col.append(ind)
        data.append(-Hz-Hss)
        for n in range(nSpins):
            # Flip a single spin from the current
            # spin configuration
            vFlip = v.copy()
            vFlip[n] = 1 - vFlip[n]
            indFlip = bitvector_to_index(vFlip)
            row.append(ind)
            col.append(indFlip)
            try:
                # Add longitudinal field contribution
                Hx = g[n]
            except TypeError:
                # Case where longitudinal field is uniform
                Hx = g
            data.append(-Hx)
    H = sparse.coo_matrix((data, (row,col)), \
                          shape=(2**nSpins,2**nSpins), \
                          dtype='float')
    # Convert to sparse CSR-matrix
    H = H.tocsr()
    return H


def get_U(Lambda, Q, t, hbar=1):
    '''
    Calculates the time-evolution operator for a system
    over a period t given the eigenvalue decomposition
    of the Hamiltonian as Lambda, Q.
    
    Arguments:
        Lambda: One dimensional numpy array of the
                eigenvalues of the Hamiltonian.
        
        Q:      Numpy matrix consisting of eigenvectors 
                of the Hamiltonian. The column Q[:, i]
                is the normalized eigenvector 
                corresponding to the eigenvalue 
                Lambda[i].
        
        t:      Float giving the time of the evolution.
        
        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.
    
    Returns:
        U:      Two-dimensional numpy matrix giving the
                time-evolution operator for the system.
    '''
    assert(hbar>0)
    powers = np.exp(-1j*Lambda*t/hbar)
    U = Q @ sparse.diags(powers) @ Q.getH()
    return U


def thermal_density_matrix(Lambda, Q, beta):
    '''
    Calculates the thermal density matrix for a system
    at inverse temperature beta given the eigenvalue 
    decomposition of the Hamiltonian as Lambda, Q.
    
    Arguments:
        Lambda: One dimensional numpy array of the
                eigenvalues of the Hamiltonian.
        
        Q:      Numpy matrix consisting of eigenvectors 
                of the Hamiltonian. The column Q[:, i]
                is the normalized eigenvector 
                corresponding to the eigenvalue 
                Lambda[i].
        
        beta:   Non-negative float or np.inf giving the 
                inverse temperature of the system in 
                units of inverse energy.
    
    Returns:
        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system.
    '''
    assert(beta>=0)
    if beta != np.inf:
        # Temperature is non-zero, all states have
        # support. Subtract off smallest eigenvalue to
        # improve performance at very low temperatures.
        powers = np.exp(-(Lambda-np.min(Lambda))*beta)
        D = powers / np.sum(powers)
        rho = Q @ sparse.diags(D) @ Q.getH()
    else:
        # Temperature is zero. Only ground state
        # has support.
        G = [1 if l==np.min(Lambda) else 0 for l in Lambda]
        D = G / np.sum(G)
        rho = Q @ sparse.diags(D) @ Q.getH()
    return rho


def calculate_A(V1, W2, V2, W3):
    '''
    Calculates the quasiprobability A(V1, W2, V2, W3)
    given projection operators V1, V2, W2, and W3.
    W2 and W3 are given in the Heisenberg picture.
    
    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator.

        W2:     A two-dimensional numpy matrix 
                representing the action of a local 
                projection operator. For time
                evolving systems, must be given in
                the Heisenberg picture.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator.

        W3:     A two-dimensional numpy matrix 
                representing the action of a local 
                projection operator. For time
                evolving systems, must be given in
                the Heisenberg picture.
                            
    Returns:
        A:      Complex float giving the evaluated
                quasiprobability.
    '''
    A = np.trace(W3 @ V2 @ W2 @ V1)
    return A


def epsilon_smooth_distribution(dist, eps, alpha):
    '''
    Heuristically smooths a probability distribution
    (dist) so as to minimize its max-entropy or
    maximize its min-entropy. The order of the Renyi
    entropy to be smoothed is given by alpha. alpha
    cannot be equal to one. eps gives the maximum 
    distance to the smoothed distribution in Total 
    Variation. This smoothing is only performed over 
    normalized distributions and does not consider 
    sub-normalized distributions.
    
    Arguments:
        dist:   A one-dimensional numpy array giving a
                normalized probability distribution
                which is to be smoothed.
        
        eps:    Smoothing factor which gives the maximum
                Total Variation distance from the initial
                distribution to the smoothed distribution.
                Must be a float between zero and one,
                exclusive.

        alpha:  Order of the Renyi entropy to be
                smoothed. Must be np.inf or a positive 
                float not equal to one.

    Returns:
        D:      eps smoothed distribution.
    '''
    assert(abs(1-np.sum(dist))<.001)
    assert(eps>0)
    assert(eps<np.sum(dist))
    # Create sorted copy of the distribution
    D = dist.copy()
    sortInds = np.argsort(D)
    Dsorted = D[sortInds]
    if alpha < 1:
        # Max-type entropy to minimize.
        # Find the largest collection of
        # low probability elements which 
        # sum to less than eps. 
        cumulDist = np.cumsum(Dsorted)
        inds = np.flatnonzero(cumulDist < eps)
        # Remove as much weight as possible
        # from these elements
        freeWeight =  np.sum(Dsorted[inds])
        Dsorted[inds] = 0
        try:
            ind = inds[-1]+1
        except IndexError:
            # No element was smaller than eps
            ind = 0
        Dsorted[ind] -= eps - freeWeight
        # Put all removed weight on the
        # maximum probability element
        Dsorted[-1] += eps
    elif alpha > 1:
        # Min-type entropy to maximize.
        # Flatten tops of the distribution
        trimWeight = np.array([[ind, np.sum(Dsorted[ind:]-floor)] \
                               for ind,floor in enumerate(Dsorted)])
        trim = trimWeight[(trimWeight[:,1]<eps)]
        try:
            # Find largest collection of peaks which can
            # be completely flattened by removing at most
            # eps weight.
            ind, freedWeight = int(trim[0,0]), trim[0,1]
            Dsorted[ind:] = Dsorted[ind]
            # Remove remaining allowance of weight from the
            # flat top of the distribution
            delta = (eps-freedWeight) / (len(Dsorted)-ind)
            Dsorted[ind:] -= delta
        except IndexError:
            # No set of peaks can be flattened by removing at 
            # most eps weight. Remove eps weight from the
            # largest peak.
            ind, freedWeight = len(Dsorted)-1, 0
            Dsorted[ind] = Dsorted[ind] - eps
        # Raise bottoms of the distribution
        fillWeight = np.array([[ind, np.sum(ceil-Dsorted[:ind+1])] \
                               for ind, ceil in enumerate(Dsorted)])
        filled = fillWeight[(fillWeight[:,1]<eps)]
        ind, spentWeight = int(filled[-1,0]), filled[-1,1]
        Dsorted[:ind+1] = Dsorted[ind]
        delta = (eps-spentWeight) / (1+ind)
        Dsorted[:ind+1] += delta
    else:
        # Smoothing isn't defined for alpha == 1.
        raise ValueError("Smoothing not defined for alpha=1")
    # Rearrange elements of the smoothed sorted distribution
    D[sortInds] = Dsorted
    return D
    

def calculate_renyi_entropy(rho, alpha, eps=0):
    '''
    Calculates the base-2 Renyi entropy of order alpha 
    for the density matrix rho. Rho may be specified
    either as a full matrix or, if diagonal, it can
    be specified by a one-dimensional array consisting
    of the diagonal elements. If a non-zero value of
    eps is given, approximates the epsilon-smoothed 
    Renyi entropy using a heuristic smoothing. eps
    gives the maximum distance to the smoothed 
    distribution in Total Variation.
    
    Arguments:
        rho:    Numpy array giving the density matrix.
                Can either be a two dimensional array
                fully specifying the matrix or, if the
                matrix is diagonal, rho can be a one-
                dimensional array giving the diagonal
                entries.
        
        alpha:  Order of the Renyi entropy to be
                calculated. Must be np.inf or a positive 
                float not equal to one.

        eps:    Smoothing factor which gives the maximum
                Total Variation distance from the initial
                distribution to the smoothed distribution.
                Must be a non-negative float less than one.
    
    Returns:
        H:      The order alpha Renyi entropy of rho as
                a float.
    '''
    assert(alpha>=0)
    assert(alpha!=1)
    if rho.ndim == 1:
        # Density matrices are non-negative.
        # Absolute value is to prevent
        # sign errors stemming from floating 
        # point error for values near zero.
        D = np.abs(rho.copy())
    else:
        D, _ = linalg.eigh(rho)
        D = np.abs(D)
    if eps > 0:
        D = epsilon_smooth_distribution(D, eps, alpha)
    if alpha == np.inf:
        H = -np.log2(np.max(D))
    else:
        # Pull out factor of largest eigenvalue to
        # improve numerical performance for very 
        # large values of alpha
        H = np.log2(np.max(D))*alpha/(1-alpha) \
            + np.log2(np.sum((D/np.max(D))**alpha))/(1-alpha)
    return H


def calculate_neumann_entropy(rho):
    '''
    Calculates the base-2 von Neumann entropy of the 
    density matrix rho. Rho may be specified either 
    as a full matrix, or if diagonal it can be 
    specified by a one-dimensional array consisting 
    of the diagonal elements.
    
    Arguments:
        rho:    Numpy array giving the density matrix.
                Can either be a two dimensional array
                fully specifying the matrix or, if the
                matrix is diagonal, rho can be a one-
                dimensional array giving the diagonal
                entries.

    Returns:
        H:      The von Neumann entropy of rho as a 
                float.
    '''
    if rho.ndim == 1:
        # Density matrices are non-negative.
        # Absolute value is to prevent
        # sign errors stemming from floating 
        # point error for values near zero.
        D = np.abs(rho.copy())
    else:
        D, _ = linalg.eigh(rho)
        D = np.abs(D)
    H = -np.sum(special.xlogy(D,D))/np.log(2)
    return H


def weak_measurement_couplings(x0, Delta, gtilde, xList, hbar=1):
    '''
    Gives the weak meausurement couplings for a
    system where the measurement is coupled to a
    particle's position and the particle is 
    initialized in a momentum-space wavepacket.
    Couplings are calculated from the density 
    functions using the simple midpoint rule where
    the midpoints are the xList values.

    Arguments:
        x0:     Reference position used to define
                the interaction between the weakly
                coupled particle and the system. The
                interaction is governed by:
                V_int = exp(-i*gtilde*(x-x0)*(\Pi^V)/hbar)
    
        Delta:  Strictly positive float giving the 
                spread of the readout particle's 
                wavepacket in momentum space.
        
        gtilde: Float giving the coupling parameter 
                between the readout particle and the 
                system.
        
        xList:  A one-dimensional numpy array of 
                length N that lists the possible 
                positions at which the readout 
                particle may be observed.
        
        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.
    
    Returns:
        pgList: A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operators.
    '''
    assert(Delta>0)
    assert(hbar>0)
    # Calculate value of density functions
    pDense = (Delta / (hbar * np.sqrt(np.pi))) \
             *np.exp(-(Delta*xList/hbar)**2)
    gDense = np.sqrt(pDense) \
             *np.expm1(-1j*(xList-x0)*gtilde/hbar)
    # Find the width associated with each xValue
    xDiffs = np.diff(xList)
    dx = np.concatenate([[xDiffs[0]], 
                         .5*(xDiffs[:-1]+xDiffs[1:]), 
                         [xDiffs[-1]]])
    # Scale couplings densities by the interval widths
    pdx = pDense * dx
    gdx = gDense * np.sqrt(dx)
    # pdx should be a nearly normalized
    # probability distribution. In case of
    # numerical errors, impose normalization
    # by hand and appropriately scale gdx.
    scaling = 1.0/np.sum(pdx)
    p = pdx * scaling
    g = gdx * np.sqrt(scaling)
    pgList = np.stack([p,g], axis=1)
    return pgList


def f_arg_W(V1, V2, pg1, pg2, W):
    '''
    For a specified strong measurement W, calculates
    the minimum over the remaining parameters of the
    argument which appears on the RHS of the
    inequality.
    
    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.
        
        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.
        
        W:      A two-dimensional numpy matrix
                representing the action of a 
                projection operator. Corresponds to a
                strong measurement. For time
                evolving systems, must be given in
                the Heisenberg picture.

    Returns:
        a:      Float giving the minimum value of the 
                argument for the specified strong 
                measurement W.
        
        aOrd:   Numpy array [LO, NLO, NNLO] of giving
                the contributions to the argument by 
                order in the small parameter g.
        
        aTerm:  Numpy array giving the contributions
                to the argument by each term on the RHS
                of the inequality.
        
        qProbs: Numpy array giving the quasiprobabilities
                [A1213, A2223, A1223].
    '''
    TrW = np.trace(W)
    # Check if V1 and V2 are the same
    if (V1!=V2).nnz==0:
        # V1 == V2
        TrWV1 = np.trace(W @ V1)
        TrWV2 = TrWV1
        A1213 = calculate_A(V1, W, V1, W)
        A2223 = A1213
        A1223 = A1213
        dV1V2 = 1
    else:
        # V1 != V2
        TrWV1 = np.trace(W @ V1)
        TrWV2 = np.trace(W @ V2)
        A1213 = calculate_A(V1, W, V1, W)
        A2223 = calculate_A(V2, W, V2, W)
        A1223 = calculate_A(V1, W, V2, W)
        dV1V2 = 0
    # Initialize the running minimum and associated
    # decompositions of the RHS.
    a = np.inf
    aOrd = np.array([])
    aTerms = np.array([])
    qProbs = np.array([A1213, A2223, A1223])
    # Need to consider all possible valid
    # combinations of weak measurement parameters.
    for p1,g1 in pg1:
        p2 = pg2[:,0]
        g2 = pg2[:,1]
        # Calculate leading order term
        t1 = np.real( -np.log2(p1*p2*TrW) )
        LO = t1
        # Calculate order g terms
        t2 = np.real( -2*np.real(g1)*TrWV1/(np.sqrt(p1)*TrW*np.log(2)) )
        t3 = np.real( -2*np.real(g2)*TrWV2/(np.sqrt(p2)*TrW*np.log(2)) )
        NLO = t2 + t3
        # Calculate order g**2 terms
        t4 = np.real( -(np.abs(g1)**2)*A1213/(p1*TrW*np.log(2)) )
        t5 = np.real( -(np.abs(g2)**2)*A2223/(p2*TrW*np.log(2)) )
        t6 = np.real( -2*np.real(g1*g2*A1223)/(np.sqrt(p1*p2)*TrW*np.log(2)) )
        t7 = np.real( -2*np.real(g1*np.conj(g2)*TrWV1)*dV1V2/(np.sqrt(p1*p2)*TrW*np.log(2)) )
        t8 = np.real( 2*((np.real(g1)*TrWV1/(np.sqrt(p1)*TrW))**2)/np.log(2) )
        t9 = np.real( 4*(np.real(g1)*TrWV1/(np.sqrt(p1)*TrW)) \
                        *(np.real(g2)*TrWV2/(np.sqrt(p2)*TrW))/np.log(2) )
        t10 = np.real( 2*((np.real(g2)*TrWV2/(np.sqrt(p2)*TrW))**2)/np.log(2) )
        NNLO = t4 + t5 + t6 + t7 + t8 + t9 + t10
        # Sum all contributions
        total = LO + NLO + NNLO
        # Check if this contains a new minimum
        if np.min(total) <= a:
            indMin = np.argmin(total)
            a = total[indMin]
            aOrd = np.array([LO[indMin], NLO[indMin], NNLO[indMin]])
            aTerm = np.array([t1[indMin], t2, t3[indMin], t4, t5[indMin], \
                              t6[indMin], t7[indMin], t8, t9[indMin], t10[indMin]])
    return a, aOrd, aTerm, qProbs


def RHS(V1, V2, pg1, pg2, Wlist):
    '''
    For a list of strong measurements (Wlist), and
    weak measurements V1 and V2 with lists of Kraus 
    operator parameters pg1 and pg2, calculates the
    RHS of the inequality.
    
    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Krauss operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Krauss operator V2.

        Wlist:  A list of two-dimensional numpy matrices 
                each representing the action of a 
                projection operator. Each matrix 
                corresponds to a strong measurement. For
                time evolving systems, must be given in
                the Heisenberg picture.

    Returns:
        f:      Float giving the RHS of the inequality.
       
        fOrd:   One-dimensional numpy array [LO, NLO, NNLO]
                giving the contributions to f by order in
                the small parameter g.

        fTerm:  Numpy array giving the contributions
                to f by each term on the RHS of the
                inequality.
        
        qProbs: Numpy array giving the quasiprobabilities
                [A1213, A2223, A1223].
    '''
    # For each W in Wlist, find the minimum of the
    # argument on the RHS of the inequality
    argList = []
    argListByOrder = []
    argListByTerm = []
    argListQuasi = []
    for W in Wlist:
        aW, aWOrd, aWTerm, aQProbs = f_arg_W(V1, V2, pg1, pg2, W)
        argList.append(aW)
        argListByOrder.append(aWOrd)
        argListByTerm.append(aWTerm)
        argListQuasi.append(aQProbs)
    # Find the minimum argument over choice of W
    indMin = np.argmin(argList)
    f = argList[indMin]
    fOrd = argListByOrder[indMin]
    fTerm = argListByTerm[indMin]
    qProbs = argListQuasi[indMin]
    return f, fOrd, fTerm, qProbs


def f_arg_strong(V1, V2, pg1, pg2, W2, W3):
    '''
    For specified strong measurements W2 and W3, 
    calculates the minimum over the remaining 
    parameters of the argument which appears on 
    the RHS of the inequality.
    
    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.
        
        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.
        
        W2:     A two-dimensional numpy matrix
                representing the action of a 
                projection operator. Corresponds to a
                strong measurement. For time
                evolving systems, must be given in
                the Heisenberg picture.

        W3:     A two-dimensional numpy matrix
                representing the action of a 
                projection operator. Corresponds to a
                strong measurement. For time
                evolving systems, must be given in
                the Heisenberg picture.

    Returns:
        f:      Float giving the minimum value of the 
                RHS for the specified strong 
                measurements W2 and W3.
        
        aOrd:   Numpy array [LO, NLO, N2LO, N3LO, N4LO]
                giving the contributions to the argument
                by order in the parameter g.
        
        aTerm:  Numpy array giving the contributions
                to the argument by each term on the RHS
                of the inequality.
        
        qProbs: Numpy array giving the quasiprobabilities
                [A1213, A2223, A1223].
    '''
    # Check V1 == V2 and W2 == W3
    if (V1!=V2).nnz==0 and np.array_equal(W2,W3):
        # V1 == V2 and W2 == W3
        dV1V2 = 1
        dW2W3 = 1
        TrW = np.trace(W2)
        TrWV1 = np.trace(W2 @ V1)
        TrWV2 = TrWV1
        A1213 = calculate_A(V1, W2, V1, W2)
        A2223 = A1213
        A1223 = A1213
    elif (V1!=V2).nnz==0 and not np.array_equal(W2,W3):
        # V1 == V2 and W2 != W3
        dV1V2 = 1
        dW2W3 = 0
        TrW = 0
        TrWV1 = 0
        TrWV2 = 0
        A1213 = calculate_A(V1, W2, V1, W3)
        A2223 = A1213
        A1223 = A1213
    elif (V1!=V2).nnz!=0 and np.array_equal(W2,W3):
        # V1 != V2 and W2 == W3
        dV1V2 = 0
        dW2W3 = 1
        TrW = np.trace(W2)
        TrWV1 = np.trace(W2 @ V1)
        TrWV2 = np.trace(W2 @ V2)
        A1213 = calculate_A(V1, W2, V1, W2)
        A2223 = calculate_A(V2, W2, V2, W2)
        A1223 = calculate_A(V1, W2, V2, W2)
    else:
        # V1 != V2 and W2 != W3
        dV1V2 = 0
        dW2W3 = 0
        TrW = 0
        TrWV1 = 0
        TrWV2 = 0
        A1213 = calculate_A(V1, W2, V1, W3)
        A2223 = calculate_A(V2, W2, V2, W3)
        A1223 = calculate_A(V1, W2, V2, W3)
    # Initialize the running minimum and associated
    # decompositions of the argument.
    f = np.inf
    aOrd = np.array([])
    aTerms = np.array([])
    qProbs = np.array([A1213, A2223, A1223])
    # Need to consider all possible valid
    # combinations of weak measurement parameters.
    for p1,g1 in pg1:
        p2 = pg2[:,0]
        g2 = pg2[:,1]
        # Calculate leading order term
        t1 = np.real(p1*p2*TrW*dW2W3)
        LO = t1
        # Calculate order g terms
        t2 = np.real(2*np.sqrt(p1)*p2*np.real(g1)*TrWV1*dW2W3)
        t3 = np.real(2*p1*np.sqrt(p2)*np.real(g2)*TrWV2*dW2W3)
        NLO = t2 + t3
        # Calculate order g**2 terms
        t4 = np.real(p2*(np.abs(g1)**2)*A1213)
        t5 = np.real(p1*(np.abs(g2)**2)*A2223)
        t6 = np.real(2*np.sqrt(p1*p2)*np.real(g1*g2*A1223))
        t7 = np.real(2*np.sqrt(p1*p2)*np.real(g1*np.conj(g2)) \
                     *TrWV1*dV1V2*dW2W3)
        N2LO = t4 + t5 + t6 + t7
        # Calculate order g**3 terms
        t8 = np.real(2*np.sqrt(p2)*(np.abs(g1)**2)*np.real(g2*A1223) \
                     *dV1V2)
        t9 = np.real(2*np.sqrt(p1)*(np.abs(g2)**2)*np.real(g1*A1223) \
                     *dV1V2)
        N3LO = t8 + t9
        # Calculate order g**4 terms
        t10 = np.real((np.abs(g1)**2)*(np.abs(g2)**2)*np.real(A1223) \
                     *dV1V2)
        N4LO = t10
        # Sum all contributions and take log of argument.
        # Truncate particularly small values.
        fullRHS = -np.log2(np.maximum(LO + NLO + N2LO + N3LO + N4LO,10**-20))
        # Check if this contains a new minimum
        if np.min(fullRHS) <= f:
            indMin = np.argmin(fullRHS)
            f = fullRHS[indMin]
            aOrd = np.array([LO[indMin], NLO[indMin], N2LO[indMin], \
                             N3LO[indMin], N4LO[indMin]])
            aTerm = np.array([t1[indMin], t2[indMin], t3[indMin], \
                              t4[indMin], t5[indMin], t6[indMin], \
                              t7[indMin], t8[indMin], t9[indMin], \
                              t10[indMin]])
    return f, aOrd, aTerm, qProbs


def RHS_strong(V1, V2, pg1, pg2, Wlist):
    '''
    For a list of strong measurements (Wlist), and
    weak measurements V1 and V2 with lists of Kraus 
    operator parameters pg1 and pg2, calculates the
    RHS of the inequality.
    
    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Krauss operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Krauss operator V2.

        Wlist:  A list of two-dimensional numpy matrices 
                each representing the action of a 
                projection operator. Each matrix 
                corresponds to a strong measurement. For
                time evolving systems, must be given in
                the Heisenberg picture.

    Returns:
        f:      Float giving the RHS of the inequality.
       
        aOrd:   Numpy array [LO, NLO, N2LO, N3LO, N4LO]
                giving the contributions to the argument
                of the RHS by order in the parameter g.
        
        aTerm:  Numpy array giving the contributions
                to the argument of the RHS by each term.
        
        qProbs: Numpy array giving the quasiprobabilities
                [A1213, A2223, A1223].
    '''
    # For each possible choice of W2 and W3 in Wlist, find the 
    # minimum of the argument on the RHS of the inequality
    fList = []
    argListByOrder = []
    argListByTerm = []
    argListQuasi = []
    Wprod = itertools.product(Wlist,Wlist)
    for W2,W3 in Wprod:
        fW, aWOrd, aWTerm, aQProbs = f_arg_strong(V1, V2, pg1, pg2, W2, W3)
        fList.append(fW)
        argListByOrder.append(aWOrd)
        argListByTerm.append(aWTerm)
        argListQuasi.append(aQProbs)
    # Find the minimum argument over choice of W2 and W3
    indMin = np.argmin(fList)
    f = fList[indMin]
    aOrd = argListByOrder[indMin]
    aTerm = argListByTerm[indMin]
    qProbs = argListQuasi[indMin]
    return f, aOrd, aTerm, qProbs


def detector_density_matrices(rho, V1, V2, pg1, pg2, Wlist):
    '''
    Calculates the density matrices for the classical
    registers of the detector under the forward and
    reverse measurement protocols. Being classical, the
    density matrices are diagonal and thus only the
    diagonal elements are returned.
    
    Arguments:
        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system.

        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional numpy matrices 
                each representing the action of a 
                projection operator. Each matrix 
                corresponds to a strong measurement. For
                time evolving systems, must be given in
                the Heisenberg picture.

    Returns:
        rhoF:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol. The detector
                is classical and thus the density matrix
                is diagonal.

        rhoR:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol. The detector
                is classical and thus the density matrix
                is diagonal.
    '''
    px1 = pg1[:,0]
    gx1 = pg1[:,1]
    px2 = pg2[:,0]
    gx2 = pg2[:,1]
    # Find density matrix for post-measurement
    # detector in forward and reverse 
    # measurement protocols. The density matrix
    # is diagonal and thus only these elements
    # need to be calculated.
    rhoFbyW = []
    rhoRbyW = []
    for W in Wlist:
        TrWrho = np.trace(W @ rho)
        TrWV1rho = np.trace(W @ V1 @ rho)
        TrWV1rhoV1 = np.trace(W @ V1 @ rho @ V1)
        TrV2WrhoW = np.trace(V2 @ W @ rho @ W)
        Df = px1*TrWrho \
             + 2*np.sqrt(px1)*np.real(gx1*TrWV1rho) \
             + (np.abs(gx1)**2)*TrWV1rhoV1
        Dr = px2*TrWrho \
             + 2*np.sqrt(px2)*np.real(gx2)*TrV2WrhoW \
             + (np.abs(gx2)**2)*TrV2WrhoW
        rhoFbyW.append(Df)
        rhoRbyW.append(Dr)
    rhoF_unnormed = np.concatenate(rhoFbyW)
    rhoR_unnormed = np.concatenate(rhoRbyW)
    # Force normalization of density matrices.
    # Note: Density matrices are diagonal and thus real.
    rhoF = np.real(rhoF_unnormed / np.sum(rhoF_unnormed))
    rhoR = np.real(rhoR_unnormed / np.sum(rhoR_unnormed))
    return rhoF, rhoR


def LHS_neumann(rhoF, rhoR):
    '''
    Calculates the von Neumann entropies which appear
    in the LHS of the inequality. 

    Arguments:
        rhoF:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol.

        rhoR:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol.
    
    Returns:
        LHSn:   Float giving the total value of the LHS.
        
        Hf:     The von Neumann entropy of the forward 
                protocol density matrix as a float.
        
        Hr:     The von Neumann entropy of the reverse 
                protocol density matrix as a float.
    '''
    Hf = calculate_neumann_entropy(rhoF)
    Hr = calculate_neumann_entropy(rhoR)
    LHSn = Hf + Hr
    return LHSn, Hf, Hr


def LHS_minmax(rhoF, rhoR, eps=0):
    '''
    Calculates the Min and Max entropies which appear
    in the LHS of the inequality. If a non-zero value
    of eps is given, instead gives the approximate
    epsilon-smoothed entropies using a heuristic 
    smoothing. eps gives the maximum distance to the 
    smoothed  distribution in Total Variation.
    
    Arguments:
        rhoF:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol.

        rhoR:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol.

        eps:    Smoothing factor which gives the maximum
                Total Variation distance from the initial
                distribution to the smoothed distribution.
                Must be a non-negative float less than one.

    Returns:
        LHSm:   Float giving the total value of the LHS.
        
        Hmin:   The Min entropy of the forward protocol
                density matrix as a float.
        
        Hmax:   The Max entropy of the reverse protocol
                density matrix as a float.
    '''
    Hmin = calculate_renyi_entropy(rhoF, np.inf, eps)
    Hmax = calculate_renyi_entropy(rhoR, .5, eps)
    LHSm = Hmin + Hmax
    return LHSm, Hmin, Hmax


def LHS(rho, V1, V2, pg1, pg2, Wlist):
    '''
    For a list of strong measurements (Wlist), weak 
    measurements (V1 and V2) with lists of Kraus 
    operator parameters (pg1 and pg2), and an initial
    density matrix (rho), calculates the LHS of the 
    inequality. Results for both the von Neumann entropy
    and the Min and Max entropies are returned.

    Arguments:
        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system.

        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional numpy matrices 
                representing the action of a local 
                projection operator. Each matrix 
                corresponds to a strong measurement. For
                time evolving systems, must be given in
                the Heisenberg picture.

    Returns:
        LHSn:   Float giving the total value of the LHS if
                the von Neumann entropies are used.
        
        LHSm:   Float giving the total value of the LHS if
                the Min and Max entropies are used.
        
        nTerms: The von Neumann entropies of the forward 
                and reverse protocols as a list [Hf, Hr].
        
        mTerms: The Min entropy of the forward protocol
                and Max entropy of the reverse protocol
                as a list [Hmin, Hmax].

        rhoF:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol. The detector
                is classical and thus the density matrix
                is diagonal.

        rhoR:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol. The detector
                is classical and thus the density matrix
                is diagonal.
    '''
    rhoF, rhoR = detector_density_matrices(rho, V1, V2, pg1, pg2, Wlist)
    LHSn, Hf, Hr = LHS_neumann(rhoF, rhoR)
    LHSm, Hmin, Hmax = LHS_minmax(rhoF, rhoR)
    nTerms = [Hf, Hr]
    mTerms = [Hmin, Hmax]
    return LHSn, LHSm, nTerms, mTerms, rhoF, rhoR


def single_time_step(t, V1, V2, pg1, pg2, Wlist, Lambda, Q, rho, hbar=1):
    '''
    Helper function which evaluates the LHS and RHS
    of the inequality at a given point in time. 
    Requires a list of strong measurements (Wlist),
    weak  measurements (V1 and V2) with lists of 
    Kraus operator parameters (pg1 and pg2), an 
    initial density matrix (rho), eigendecomposition 
    of the Hamiltonian (Lambda, Q), and a time (t).

    Arguments:
        t:      Time over which the system is allowed
                to evolve.
        
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional sparse 
                CSR-matrices representing the action of 
                local projection operators. Each matrix 
                corresponds to a strong measurement
                performed on the system at time t=0.

        Lambda: One dimensional numpy array of the
                eigenvalues of the Hamiltonian.
        
        Q:      Numpy matrix consisting of eigenvectors 
                of the Hamiltonian. The column Q[:, i]
                is the normalized eigenvector 
                corresponding to the eigenvalue 
                Lambda[i].

        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system at time
                t=0.

        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.
        
    Returns:
        Neu:   Float giving the total value of the LHS if
                the von Neumann entropies are used.
        
        MinMax:   Float giving the total value of the LHS if
                the Min and Max entropies are used.
        
        Nterms: The von Neumann entropies of the forward 
                and reverse protocols as a list [Hf, Hr].
        
        Mterms: The Min entropy of the forward protocol
                and Max entropy of the reverse protocol
                as a list [Hmin, Hmax].

        rhoF:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol. The detector
                is classical and thus the density matrix
                is diagonal.

        rhoR:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol. The detector
                is classical and thus the density matrix
                is diagonal.

        f:      Float giving the RHS of the inequality.
       
        fOrd:   One-dimensional numpy array [LO, NLO, NNLO]
                giving the contributions to f by order in
                the small parameter g.

        fTerm:  Numpy array giving the contributions
                to f by each term on the RHS of the
                inequality.
        
        qProbs: Numpy array giving the quasiprobabilities
                [A1213, A2223, A1223].
    '''
    assert(hbar>0)
    U = get_U(Lambda,Q,t,hbar)
    WevoList = [U.getH() @ W @ U for W in Wlist]
    Neu, MinMax, Nterms, Mterms, rhoF, rhoR = LHS(rho, V1, V2, pg1, pg2, WevoList)
    f, fOrd, fTerm, qProbs = RHS(V1, V2, pg1, pg2, WevoList)
    return Neu, MinMax, Nterms, Mterms, rhoF, rhoR, f, fOrd, fTerm, qProbs


def single_time_step_strong(t, V1, V2, pg1, pg2, Wlist, Lambda, Q, rho, hbar=1):
    '''
    Helper function which evaluates the LHS and RHS
    of the inequality at a given point in time. 
    Requires a list of strong measurements (Wlist),
    weak  measurements (V1 and V2) with lists of 
    Kraus operator parameters (pg1 and pg2), an 
    initial density matrix (rho), eigendecomposition 
    of the Hamiltonian (Lambda, Q), and a time (t).

    Arguments:
        t:      Time over which the system is allowed
                to evolve.
        
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional sparse 
                CSR-matrices representing the action of 
                local projection operators. Each matrix 
                corresponds to a strong measurement
                performed on the system at time t=0.

        Lambda: One dimensional numpy array of the
                eigenvalues of the Hamiltonian.
        
        Q:      Numpy matrix consisting of eigenvectors 
                of the Hamiltonian. The column Q[:, i]
                is the normalized eigenvector 
                corresponding to the eigenvalue 
                Lambda[i].

        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system at time
                t=0.

        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.
        
    Returns:
        Neu:   Float giving the total value of the LHS if
                the von Neumann entropies are used.
        
        MinMax:   Float giving the total value of the LHS if
                the Min and Max entropies are used.
        
        Nterms: The von Neumann entropies of the forward 
                and reverse protocols as a list [Hf, Hr].
        
        Mterms: The Min entropy of the forward protocol
                and Max entropy of the reverse protocol
                as a list [Hmin, Hmax].

        rhoF:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol. The detector
                is classical and thus the density matrix
                is diagonal.

        rhoR:   One-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol. The detector
                is classical and thus the density matrix
                is diagonal.

        f:      Float giving the RHS of the inequality.
       
        aOrd:   Numpy array [LO, NLO, N2LO, N3LO, N4LO]
                giving the contributions to the argument
                of the RHS by order in the parameter g.
        
        aTerm:  Numpy array giving the contributions
                to the argument of the RHS by each term.
        
        qProbs: Numpy array giving the quasiprobabilities
                [A1213, A2223, A1223].
    '''
    assert(hbar>0)
    U = get_U(Lambda,Q,t,hbar)
    WevoList = [U.getH() @ W @ U for W in Wlist]
    Neu, MinMax, Nterms, Mterms, rhoF, rhoR = LHS(rho, V1, V2, pg1, pg2, WevoList)
    f, aOrd, aTerm, qProbs = RHS_strong(V1, V2, pg1, pg2, WevoList)
    return Neu, MinMax, Nterms, Mterms, rhoF, rhoR, f, aOrd, aTerm, qProbs


def simulation(V1, V2, pg1, pg2, Wlist, H, rho, tList, hbar=1, nWorks=1):
    '''
    For a list of points in time (tList), calculates
    both sides of the inequality as functions of
    time. Requires a list of strong measurements 
    (Wlist), weak measurements (V1 and V2) with lists 
    of Kraus operator parameters (pg1 and pg2), a
    Hamiltonian (H), and an initial density matrix 
    (rho). Each time in tList is calculated in parallel
    and the number of parallel workers is set by
    nWorks.

    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional sparse 
                CSR-matrices representing the action of 
                local projection operators. Each matrix 
                corresponds to a strong measurement
                performed on the system at time t=0.

        H:      Hamiltonian governing the evolution of
                the system. May be given either as a
                sparse matrix or numpy matrix.

        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system at time
                t=0.

        tList:  List of times at which the inequality is
                to be evaluated at.

        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.
        
        nWorks: Positive integer denoting the number of 
                workers to be used for parallel processing.
                Default value is 1.

    Returns:
        LHSneu: One dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the von Neumann entropies are used.
        
        LHSmm:  One-dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the Min and Max entropies are used.

        LHSneuTerms: Two-dimensional numpy array giving
                     the von Neumann entropies of the forward 
                     and reverse protocols as a function of
                     time with columns [Hf, Hr].

        LHSmmTerms:  Two-dimensional numpy array giving
                     the Min entropy of the forward protocol
                     and Max entropy of the reverse protocol
                     as a function of time with columns
                     [Hmin, Hmax].

        rhoF:   Two-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol as a function
                of time. The detector is classical and thus 
                the density matrix is diagonal.

        rhoR:   Two-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol as a function
                of time. The detector is classical and thus 
                the density matrix is diagonal.

        f:      One-dimensional numpy array giving the RHS of 
                the inequality as a function of time.
       
        fOrd:   Two-dimensional numpy array with columns
                [LO, NLO, NNLO] giving the contributions to f 
                by order in the small parameter g as a function
                of time.

        fTerm:  Two-dimensional numpy array with columns giving 
                the contributions to f by each term on the
                RHS of the inequality as a function of time.

        qProbs: Two-dimensional numpy array giving the 
                quasiprobabilities as a function of time with
                columns [A1213, A2223, A1223].
    '''
    assert(hbar>0)
    assert(isinstance(nWorks, int))
    # If Hamiltonian is given as a sparse matrix, 
    # convert to dense matrix.
    if sparse.isspmatrix(H):
        H = H.todense()
    # Decompose Hamiltonian
    Lambda, Q = linalg.eigh(H)
    # At each specified time-step, calculate both 
    # sides of the inequality
    argList = ( (t, V1, V2, pg1, pg2, Wlist, Lambda, Q, rho, hbar) for t in tList)
    pool = mp.Pool(processes=nWorks)
    results = pool.starmap(single_time_step, argList)
    pool.close()
    pool.join()
    # Consolidate results as numpy arrays and return
    LHSneu = np.array([r[0] for r in results])
    LHSmm = np.array([r[1] for r in results])
    LHSneuTerms = np.array([r[2] for r in results])
    LHSmmTerms = np.array([r[3] for r in results])
    rhoF = np.array([r[4] for r in results])
    rhoR = np.array([r[5] for r in results])
    f = np.array([r[6] for r in results])
    fOrd = np.array([r[7] for r in results])
    fTerm = np.array([r[8] for r in results])
    qProbs = np.array([r[9] for r in results])
    return LHSneu, LHSmm, LHSneuTerms, LHSmmTerms, rhoF, rhoR, f, fOrd, fTerm, qProbs


def simulation_strong(V1, V2, pg1, pg2, Wlist, H, rho, tList, hbar=1, nWorks=1):
    '''
    For a list of points in time (tList), calculates
    both sides of the inequality as functions of
    time. Requires a list of strong measurements 
    (Wlist), weak measurements (V1 and V2) with lists 
    of Kraus operator parameters (pg1 and pg2), a
    Hamiltonian (H), and an initial density matrix 
    (rho). Each time in tList is calculated in parallel
    and the number of parallel workers is set by
    nWorks.

    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional sparse 
                CSR-matrices representing the action of 
                local projection operators. Each matrix 
                corresponds to a strong measurement
                performed on the system at time t=0.

        H:      Hamiltonian governing the evolution of
                the system. May be given either as a
                sparse matrix or numpy matrix.

        rho:    Two-dimensional numpy matrix giving the
                density matrix for the system at time
                t=0.

        tList:  List of times at which the inequality is
                to be evaluated at.

        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.
        
        nWorks: Positive integer denoting the number of 
                workers to be used for parallel processing.
                Default value is 1.

    Returns:
        LHSneu: One dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the von Neumann entropies are used.
        
        LHSmm:  One-dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the Min and Max entropies are used.

        LHSneuTerms: Two-dimensional numpy array giving
                     the von Neumann entropies of the forward 
                     and reverse protocols as a function of
                     time with columns [Hf, Hr].

        LHSmmTerms:  Two-dimensional numpy array giving
                     the Min entropy of the forward protocol
                     and Max entropy of the reverse protocol
                     as a function of time with columns
                     [Hmin, Hmax].

        rhoF:   Two-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol as a function
                of time. The detector is classical and thus 
                the density matrix is diagonal.

        rhoR:   Two-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol as a function
                of time. The detector is classical and thus 
                the density matrix is diagonal.

        f:      One-dimensional numpy array giving the RHS of 
                the inequality as a function of time.
       
        aOrd:   Two-dimensional numpy array with columns
                [LO, NLO, N2LO, N3LO, N4LO] giving the 
                contributions to the argument of the RHS by 
                order in the parameter g as a function of time.

        aTerm:  Two-dimensional numpy array with columns giving 
                the contributions to the argument of the RHS by 
                each term as a function of time.

        qProbs: Two-dimensional numpy array giving the 
                quasiprobabilities as a function of time with
                columns [A1213, A2223, A1223].
    '''
    assert(hbar>0)
    assert(isinstance(nWorks, int))
    # If Hamiltonian is given as a sparse matrix, 
    # convert to dense matrix.
    if sparse.isspmatrix(H):
        H = H.todense()
    # Decompose Hamiltonian
    Lambda, Q = linalg.eigh(H)
    # At each specified time-step, calculate both 
    # sides of the inequality
    argList = ( (t, V1, V2, pg1, pg2, Wlist, Lambda, Q, rho, hbar) for t in tList)
    pool = mp.Pool(processes=nWorks)
    results = pool.starmap(single_time_step_strong, argList)
    pool.close()
    pool.join()
    # Consolidate results as numpy arrays and return
    LHSneu = np.array([r[0] for r in results])
    LHSmm = np.array([r[1] for r in results])
    LHSneuTerms = np.array([r[2] for r in results])
    LHSmmTerms = np.array([r[3] for r in results])
    rhoF = np.array([r[4] for r in results])
    rhoR = np.array([r[5] for r in results])
    f = np.array([r[6] for r in results])
    aOrd = np.array([r[7] for r in results])
    aTerm = np.array([r[8] for r in results])
    qProbs = np.array([r[9] for r in results])
    return LHSneu, LHSmm, LHSneuTerms, LHSmmTerms, rhoF, rhoR, f, aOrd, aTerm, qProbs


def visualize_weak_couplings(pg, xList):
    '''
    Create plots visualizing the weak coupling
    parameters. The first plot visualizes sqrt(px),
    Re[gx], and Im[gx] as functions of x. The
    second plot visualizes the ratio |gx|/sqrt(px).

    Arguments:
        pg:     A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operators.

        xList:  A one-dimensional numpy array of 
                length N that lists the possible 
                positions at which the readout 
                particle may be observed.

    Returns:
        None
    '''
    # Plot Sqrt(Px), Re[gx], and Im[gx].
    plt.plot(xList, np.sqrt(np.real(pg[:,0])), label='sqrt(p)')
    plt.plot(xList, np.real(pg[:,1]), label='g (real)')
    plt.plot(xList, np.imag(pg[:,1]), label='g (imag)')
    plt.title("Px and gx")
    plt.xlabel("Readout Position")
    plt.ylabel("Sqrt(Probability)")
    plt.legend()
    plt.show()
    # Plot the ratio |gx|/Sqrt(Px).
    # Small constant eps to prevent the edge case of 0/0.
    eps = 10**-10
    plt.plot(xList, np.abs(pg[:,1])/(np.sqrt(np.real(pg[:,0]))+eps), label='|g|/sqrt(p)')
    plt.ylim(0)
    plt.title("Perturbativity")
    plt.xlabel("Readout Position")
    plt.ylabel("Ratio")
    plt.legend()
    plt.show()
    return


def visualize_inequality(tList, LHSneu, LHSmm, f, fOrd, fTerm, qProbs):
    '''
    Visualizes the LHS and RHS of the inequality as a
    function of time. Also generates visualizations of
    the contributions to the RHS by order in g and of
    the OTOC quasiprobabilities.

    Arguments:
        tList:  List of times at which the inequality was
                evaluated at.

        LHSneu: One dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the von Neumann entropies are used.
        
        LHSmm:  One-dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the Min and Max entropies are used.

        f:      One-dimensional numpy array giving the RHS of 
                the inequality as a function of time.
       
        fOrd:   Two-dimensional numpy array with columns
                [LO, NLO, NNLO] giving the contributions to f 
                by order in the small parameter g as a function
                of time.

        fTerm:  Two-dimensional numpy array with columns giving 
                the contributions to f by each term on the
                RHS of the inequality as a function of time.

        qProbs: Two-dimensional numpy array giving the 
                quasiprobabilities as a function of time with
                columns [A1213, A2223, A1223].

    Returns:
        None
    '''
    # Plot both sides of the inequality
    plt.plot(tList, LHSmm, label='MinMax')
    plt.plot(tList, LHSneu, label='Neumann')
    plt.plot(tList, f, label='RHS')
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("Inequality")
    plt.legend()
    plt.show()
    # Plot the RHS by order in g
    plt.plot(tList, f, label='Total')
    plt.plot(tList, fOrd[:,0], label='LO')
    plt.plot(tList, fOrd[:,1], label='NLO')
    plt.plot(tList, fOrd[:,2], label='NNLO')
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("RHS by order in g")
    plt.legend()
    plt.show()
    # Plot the contributions to the RHS by
    # terms which include quasiprobabilities
    plt.plot(tList, fOrd[:,2], label='NNLO')
    plt.plot(tList, fTerm[:,3], label='Term-A1213')
    plt.plot(tList, fTerm[:,4], label='Term-A2223')
    plt.plot(tList, fTerm[:,5], label='Term-A1223')
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("Quasiprobability Contributions")
    plt.legend()
    plt.show()
    # Plot the quasiprobabilities
    A1213 = qProbs[:,0]
    A2223 = qProbs[:,1]
    A1223 = qProbs[:,2]
    # Plot quasiprobability A1213
    plt.plot(tList, np.real(A1213), label='Re[A1213]')
    plt.plot(tList, np.imag(A1213), label='Im[A1213]')
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("(Argmin) Quasiprobability A1213")
    plt.legend()
    plt.show()
    # Plot quasiprobability A2223
    plt.plot(tList, np.real(A2223), label='Re[A2223]')
    plt.plot(tList, np.imag(A2223), label='Im[A2223]')
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("(Argmin) Quasiprobability A2223")
    plt.legend()
    plt.show()
    # Plot quasiprobability A1223
    plt.plot(tList, np.real(A1223), label='Re[A1223]')
    plt.plot(tList, np.imag(A1223), label='Im[A2223]')
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("(Argmin) Quasiprobability A1223")
    plt.legend()
    plt.show()
    return


def visualize_inequality_strong(tList, LHSneu, LHSmm, f, aOrd, aTerm, qProbs):
    '''
    Visualizes the LHS and RHS of the inequality as a
    function of time. Also generates visualizations of
    the contributions to the RHS by order in g and of
    the OTOC quasiprobabilities.

    Arguments:
        tList:  List of times at which the inequality was
                evaluated at.

        LHSneu: One dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the von Neumann entropies are used.
        
        LHSmm:  One-dimensional numpy array giving the 
                total value of the LHS as a function of
                time if the Min and Max entropies are used.
    
        f:      One-dimensional numpy array giving the RHS of 
                the inequality as a function of time.
       
        aOrd:   Two-dimensional numpy array with columns
                [LO, NLO, N2LO, N3LO, N4LO] giving the 
                contributions to the argument of the RHS by 
                order in the parameter g as a function of time.

        aTerm:  Two-dimensional numpy array with columns giving 
                the contributions to the argument of the RHS by 
                each term as a function of time.

        qProbs: Two-dimensional numpy array giving the 
                quasiprobabilities as a function of time with
                columns [A1213, A2223, A1223].

    Returns:
        None
    '''
    # Plot both sides of the inequality
    plt.plot(tList, LHSmm, label='MinMax')
    plt.plot(tList, LHSneu, label='Neumann')
    plt.plot(tList, f, label='RHS')
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("Inequality")
    plt.legend()
    plt.show()
    # Plot the RHS by order in g
    plt.plot(tList, np.sum(aOrd, axis=1), label='Total')
    plt.plot(tList, aOrd[:,0], label='LO')
    plt.plot(tList, aOrd[:,1], label='NLO')
    plt.plot(tList, aOrd[:,2], label='N2LO')
    plt.plot(tList, aOrd[:,3], label='N3LO')
    plt.plot(tList, aOrd[:,4], label='N4LO')
    plt.xlabel("Time")
    plt.ylabel("Argument")
    plt.title("RHS by order in g")
    plt.legend()
    plt.show()
    # Plot the quasiprobabilities
    A1213 = qProbs[:,0]
    A2223 = qProbs[:,1]
    A1223 = qProbs[:,2]
    # Plot quasiprobability A1213
    plt.plot(tList, np.real(A1213), label='Re[A1213]')
    plt.plot(tList, np.imag(A1213), label='Im[A1213]')
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("(Argmin) Quasiprobability A1213")
    plt.legend()
    plt.show()
    # Plot quasiprobability A2223
    plt.plot(tList, np.real(A2223), label='Re[A2223]')
    plt.plot(tList, np.imag(A2223), label='Im[A2223]')
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("(Argmin) Quasiprobability A2223")
    plt.legend()
    plt.show()
    # Plot quasiprobability A1223
    plt.plot(tList, np.real(A1223), label='Re[A1223]')
    plt.plot(tList, np.imag(A1223), label='Im[A2223]')
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("(Argmin) Quasiprobability A1223")
    plt.legend()
    plt.show()
    return


def visualize_detector_state(xList, rhoD, titleS=None):
    '''
    Visualizes the LHS and RHS of the inequality as a
    function of time. Also generates visualizations of
    the contributions to the RHS by order in g and of
    the OTOC quasiprobabilities.

    Arguments:
        xList:  A one-dimensional numpy array that 
                lists the possible positions at which 
                the readout particle may be observed.

        rhoD:   One-dimensional numpy array giving the
                diagonal elements of the detector's 
                density matrix or a two-dimensional 
                numpy array giving these elements as a
                function of time. The detector is 
                classical and thus the density matrix
                is diagonal. If the density matrix is
                given as a function of time, the
                average probability over time is used
                in the visualization.
        
        titleS: String giving the title for the plot.
                Optional, default value of None will
                display no title.
    
    Returns:
        None
    '''
    if rhoD.ndim != 1:
        # Average detector probabilities if
        # given as a function of time.
        rhoD = np.mean(rhoD, axis=0)
    # Find number of readout positions for the particle
    # and number of possible strong measurements
    nReadout = len(xList)
    nStrong = len(rhoD)/nReadout
    # Split the density matrix into components based on
    # the strong measurement performed.
    splitRho = np.split(rhoD, nStrong)
    # Plot all the components
    for ind, subRho in enumerate(splitRho):
        plt.plot(xList, subRho, label='W-'+repr(ind))
    plt.xlabel('Readout Position')
    plt.ylabel('Probability')
    if titleS is not None:
        plt.title(titleS)
    plt.legend()
    plt.show()
    return


def visualize_smoothed_inequality(tList, rhoF, rhoR, f, eps):
    '''
    Visualizes the original and epsilon-smoothed inequalities.
    
    Arguments:
        tList:  List of times at which the inequality was
                evaluated at.

        rhoF:   Two-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the forward protocol as a function
                of time. The detector is classical and thus 
                the density matrix is diagonal.

        rhoR:   Two-dimensional numpy array giving the
                diagonal elements of the detector's density
                matrix in the reverse protocol as a function
                of time. The detector is classical and thus 
                the density matrix is diagonal.

        f:      One-dimensional numpy array giving the RHS of 
                the inequality as a function of time.
           
        eps:    Smoothing factor which gives the maximum
                Total Variation distance from the initial
                distribution to the smoothed distribution.
                Must be a non-negative float less than one.

    Returns:
        Hmin:   One-dimensional numpy array giving the
                min-entropy of the forward process as a
                function of time.

        Hmax:   One-dimensional numpy array giving the
                max-entropy of the reverse process as a
                function of time.

        Hmineps: One-dimensional numpy array giving the
                 smoothed min-entropy of the forward 
                 process as a function of time.

        Hmaxeps: One-dimensional numpy array giving the
                 smoothed max-entropy of the reverse
                 process as a function of time.
    '''
    nt = len(tList)
    Hmin = np.zeros(nt)
    Hmax = np.zeros(nt)
    Hmineps = np.zeros(nt)
    Hmaxeps = np.zeros(nt)
    for ind in range(nt):
        rhoFt = rhoF[ind,:]
        rhoRt = rhoR[ind,:]
        _, Hmin[ind], Hmax[ind] = LHS_minmax(rhoFt, rhoRt)
        _, Hmineps[ind], Hmaxeps[ind] = LHS_minmax(rhoFt, rhoRt, eps)
    # Plot smoothed and original inequalities
    plt.plot(tList, Hmin+Hmax, label='MinMax')
    plt.plot(tList, Hmineps+Hmaxeps, label='eps-MinMax')
    plt.plot(tList, f, label='RHS')
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("Original and Smoothed Inequalities")
    plt.legend()
    plt.show()
    # Plot change in LHS
    plt.plot(tList, Hmin + Hmax - Hmineps - Hmaxeps, label="LHS - LHS_eps")
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("Change in LHS")
    plt.legend()
    plt.show()
    # Plot Min and Max entropies
    plt.plot(tList, Hmin, label='Hmin')
    plt.plot(tList, Hmineps, label='Hmin_eps')
    plt.plot(tList, Hmax, label='Hmax')
    plt.plot(tList, Hmaxeps, label='Hmax_eps')
    plt.xlabel("Time")
    plt.ylabel("Bits")
    plt.title("Min and Max Entropies")
    plt.legend()
    plt.show()
    return Hmin, Hmax, Hmineps, Hmaxeps


def create_all_quasiprobs(V1, V2, pg1, pg2, Wlist, H, tList, hbar=1):
    '''
    For a list of points in time (tList), calculates
    the quasiprobabilities A1213, A2223, and A1213 as
    functions of the choice of W and as functions of
    time. Requires a list of strong measurements 
    (Wlist), weak measurements (V1 and V2) with lists 
    of Kraus operator parameters (pg1 and pg2), and a
    Hamiltonian (H).

    Arguments:
        V1:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        V2:     A two-dimensional sparse CSR-matrix 
                representing the action of a local 
                projection operator. Corresponds to a
                weak measurement.

        pg1:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V1.

        pg2:    A two-dimensional numpy array with
                shape (N x 2) and columns [px,gx]
                listing the values for the weak 
                measurement Kraus operator V2.

        Wlist:  A list of two-dimensional sparse 
                CSR-matrices representing the action of 
                local projection operators. Each matrix 
                corresponds to a strong measurement
                performed on the system at time t=0.

        H:      Hamiltonian governing the evolution of
                the system. May be given either as a
                sparse matrix or numpy matrix.

        tList:  List of times at which the 
                quasiprobabilities are to be evaluated at.

        hbar:   Positive float giving the value of the 
                reduced Planck constant. By default, 
                units chosen so that hbar = 1.

    Returns:
        tA1213: Two-dimensional numpy array giving the 
                quasiprobability A1213 as a function of time 
                with columns corresponding to the choice of 
                strong measurement from Wlist.

        tA2223: Two-dimensional numpy array giving the 
                quasiprobability A2223 as a function of time 
                with columns corresponding to the choice of 
                strong measurement from Wlist.

        tA1223: Two-dimensional numpy array giving the 
                quasiprobability A1223 as a function of time 
                with columns corresponding to the choice of 
                strong measurement from Wlist.
    '''
    assert(hbar>0)
    # If Hamiltonian is given as a sparse matrix, 
    # convert to dense matrix.
    if sparse.isspmatrix(H):
        H = H.todense()
    # Decompose Hamiltonian
    Lambda, Q = linalg.eigh(H)
    # Initialize arrays to store quasiprobabilities
    tA1213 = []
    tA2223 = []
    tA1223 = []
    if (V1!=V2).nnz==0:
        # V1 == V2
        for t in tList:
            U = get_U(Lambda,Q,t,hbar)
            WevoList = [U.getH() @ W @ U for W in Wlist]
            tA1213.append([calculate_A(V1, W, V1, W) for W in WevoList])
        tA1213 = np.array(tA1213)
        tA2223 = np.copy(tA1213)
        tA1223 = np.copy(tA1213)
    else:
        # V1 != V2
        for t in tList:
            U = get_U(Lambda,Q,t,hbar)
            WevoList = [U.getH() @ W @ U for W in Wlist]
            tA1213.append([calculate_A(V1, W, V1, W) for W in WevoList])
            tA2223.append([calculate_A(V2, W, V2, W) for W in WevoList])
            tA1223.append([calculate_A(V1, W, V2, W) for W in WevoList])
        tA1213 = np.array(tA1213)
        tA2223 = np.array(tA2223)
        tA1223 = np.array(tA1223)
    return tA1213, tA2223, tA1223


def visualize_all_quasiprobs(tList, tA1213, tA2223, tA1223):
    '''
    For a list of points in time (tList), and 
    two-dimensional numpy arrays tA1213, tA2223, tA1223
    giving the quasiprobabilities as a function of time,
    plots all possible quasiprobabilities for the system.

    Arguments:
        tList:  List of times at which the 
                quasiprobabilities were evaluated.
        
        tA1213: Two-dimensional numpy array giving the 
                quasiprobability A1213 as a function of time 
                with columns corresponding to the choice of 
                strong measurement from Wlist.

        tA2223: Two-dimensional numpy array giving the 
                quasiprobability A2223 as a function of time 
                with columns corresponding to the choice of 
                strong measurement from Wlist.

        tA1223: Two-dimensional numpy array giving the 
                quasiprobability A1223 as a function of time 
                with columns corresponding to the choice of 
                strong measurement from Wlist.

    Returns:
        None
    '''
    # Check that array dimensions match
    assert(tA1213.shape == tA1223.shape)
    assert(tA2223.shape == tA1223.shape)
    assert(len(tList) == tA1223.shape[0])
    # Number of possible strong measurements considered
    N = tA1213.shape[1]
    # Plot the real and imaginary components of A1213
    for ind,c in enumerate(np.linspace(0,1,N)):
        q = tA1213[:,ind]
        plt.plot(tList, np.real(q), color=(1-c,.2,c))
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.ylim(0)
    plt.title("Real[A1213]")
    plt.show()
    for ind,c in enumerate(np.linspace(0,1,N)):
        q = tA1213[:,ind]
        plt.plot(tList, np.imag(q), color=(1-c,.2,c))
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("Imag[A1213]")
    plt.show()
    # Plot the real and imaginary components of A2223
    for ind,c in enumerate(np.linspace(0,1,N)):
        q = tA2223[:,ind]
        plt.plot(tList, np.real(q), color=(1-c,.2,c))
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.ylim(0)
    plt.title("Real[A2223]")
    plt.show()
    for ind,c in enumerate(np.linspace(0,1,N)):
        q = tA2223[:,ind]
        plt.plot(tList, np.imag(q), color=(1-c,.2,c))
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("Imag[A2223]")
    plt.show()
    # Plot the real and imaginary components of A1223
    for ind,c in enumerate(np.linspace(0,1,N)):
        q = tA1223[:,ind]
        plt.plot(tList, np.real(q), color=(1-c,.2,c))
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.ylim(0)
    plt.title("Real[A1223]")
    plt.show()
    for ind,c in enumerate(np.linspace(0,1,N)):
        q = tA1223[:,ind]
        plt.plot(tList, np.imag(q), color=(1-c,.2,c))
    plt.xlabel("Time")
    plt.ylabel("Unnormalized Quasiprobability")
    plt.title("Imag[A1223]")
    plt.show()
    return

