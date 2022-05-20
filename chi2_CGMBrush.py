import numpy as np
from FRB_utils import weighting_function
from CGMBrush_utils import *


def calc_chi2_model(Mhalos, b2Rvirs, DMs, alpha, beta, weighted_meanDM, model_name,
                    b2Rvir_bin_edges=[0.25,0.5,1.0,1.5,2.0]):
    '''
    Input the Mhalos, b2Rvirs, DMs arrays for the FRBs that have foreground CGM intersection.
    alpha, beta are the inputs for the exp weighting function.
    weighted_meanDM is the weighted-mean DM of the whole sample, i.e. no DM excess due to CGM.
    model_name is name of the model to be evaluated.  If input a number, interpret as const DM excess.
    We calculate each chi2 in a radial bin and return the chi2 array.  Also returned is the array of the number of FRBs in radial bins, and the DM excess predicted by the model.
    '''
    # get the model predicted DM excess for each FRB
    model_DMexcs = get_model_DMexc(Mhalos, b2Rvirs, model_name)

    # calculate chi2 in each radial bin
    chi2s_bin = np.zeros(len(b2Rvir_bin_edges)-1)
    n_frbs_bin = np.zeros(len(b2Rvir_bin_edges)-1, dtype=int)
    for i in range(len(b2Rvir_bin_edges)-1):
        ii = (b2Rvirs > b2Rvir_bin_edges[i]) & (b2Rvirs < b2Rvir_bin_edges[i+1])
        n_frbs_bin[i] = ii.sum()
        if n_frbs_bin[i] == 0:
            continue
        DMs_ = DMs[ii] - model_DMexcs[ii]
        weighted_meanDM_bin = np.average(DMs_, weights=weighting_function(DMs_, alpha, beta))
        chi2s_bin[i] = (weighted_meanDM_bin - weighted_meanDM)**2

    return chi2s_bin, n_frbs_bin, model_DMexcs


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

