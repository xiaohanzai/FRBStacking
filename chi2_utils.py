import numpy as np
from FRB_utils import weighting_function


def calc_chi2_normalization(n_frb, DMs, alpha, beta):
    '''
    Normalization of the chi2, calculated based on the number of FRBs and the weighting function parameters alpha, beta.
    DMs should be the full array of DM values, i.e. without CGM intersection.
    '''
    if n_frb == 0: # no FRBs in radial bin
        return 1e6
    ws = weighting_function(DMs, alpha, beta)
    weighted_meanDM = np.average(DMs, weights=ws)
    N = 10000
    rst = 0.
    for i in range(N):
        inds = np.random.choice(len(DMs), n_frb, replace=True)
        rst += (np.average(DMs[inds], weights=ws[inds]) - weighted_meanDM)**2
    return rst/N


def calc_marginalized_chi2s(chi2s_bin):
    '''
    Calculate the likelihood by marginalizing over localization error.  Return effective chi2 = -2*ln(L).
    chi2s_bin should be the (normalized) Nxn_radial_bin array returned from Monte Carlo sampling over the localization contour.
    '''
    Ls = np.mean(np.exp(-chi2s_bin/2.), axis=0) # likelihood in each radial bin
    L_tot = np.mean(np.exp(-chi2s_bin.sum(axis=1)/2.)) # summed over radial bins
    return -2*np.log(Ls), -2*np.log(L_tot)

