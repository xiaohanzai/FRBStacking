import numpy as np
from astropy.table import Table
from cfod import catalog


# names of localized non-repeaters before 2020
localized_list = ['20180924B', '20181112A', '20190102C', '20190523A', '20190608B',
                  '20190611B', '20190614D', '20190714A', '20191001A', '20191228A']


# def load_FRBs(gb_cut=5.):
#     cat_frb_ = catalog.as_dataframe()
#     inds_frb = np.where((cat_frb['repeater_name']=='-9999') & (cat_frb['sub_num']==0) &
#                         (np.abs(cat_frb['gb'])>gb_cut))[0]
#     cat_frb = Table()
#     for key in cat_frb_.columns:
#         cat_frb[key] = cat_frb_[key][inds_frb].values
#     return cat_frb


def weighting_function(DMarr, alpha, beta):
    argvec = np.asarray((DMarr/alpha)**beta)
    argvec[argvec>100] = 100  #because this goes in an exp, to safegaurd against overflow
    return np.exp(-argvec)


def gen_mock_DMs(DMs, DMmin=80, DMmax=3000, n_DM=100000):
    '''
    Given an array of DM values, construct mock catalog of n_DM values that has the same PDF of DM.
    '''
    # getting the histogram
    counts, bin_edges = np.histogram(np.log10(DMs), bins=200)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
    # finding the PDF of the histogram using count values
    pdf = counts / np.sum(counts)  
    # using numpy np.cumsum to calculate the CDF
    cdf = np.cumsum(pdf)

    CDF = lambda DM: np.interp(np.log10(DM), bin_centers, cdf)

    n_P = 1000 #how finely we sample in logspace DM
    DMvals = np.logspace(np.log10(DMmin), np.log10(DMmax), n_P)  #a grid of DM values
    CPvals = CDF(DMvals) #the cummulative PDF evaluated on this grid

    # this creates the sample
    mockDMs = np.interp(np.random.rand(n_DM), CPvals, DMvals)

    return mockDMs


def calc_DMexc_distribution(DMs, ws, n_frb, n_sample=10000):
    '''
    Given an array of DM values and their corresponding weights, sample n_frb values from the array n_sample times
      to get the distribution of the difference in the weighted-average DM.
    '''
    meanDM = np.average(DMs, weights=ws)
    inds = np.random.choice(len(DMs), (n_sample, n_frb))
    diff_meanDMs = np.average(DMs[inds], weights=ws[inds], axis=1) - meanDM
    return diff_meanDMs


def calc_DMexc_const(DMs1, alpha, beta, meanDM0):
    '''
    DMs1 is assumed to have a const DM excess.  meanDM0 is the weighted-average DM of the original sample.
    Iterate until convergence on DMexc.
    '''
    DMexc = np.average(DMs1, weights=weighting_function(DMs1, alpha, beta)) - meanDM0
    DMexc_new = np.average(DMs1, weights=weighting_function(DMs1-DMexc, alpha, beta)) - meanDM0
    n_iter = 0
    while np.abs(DMexc_new - DMexc) > 1e-3:
        DMexc = DMexc_new
        DMexc_new = np.average(DMs1, weights=weighting_function(DMs1-DMexc, alpha, beta)) - meanDM0
        n_iter += 1
        if n_iter > 1000:
            print('not converged?')
            break
    return DMexc
