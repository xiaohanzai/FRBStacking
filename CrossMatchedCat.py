import numpy as np
import scipy.stats
import random
from gal_utils import *
from FRB_utils import *
from crossmatch_utils import *
from plot_utils import *
from chi2_CGMBrush import *
from CGMBrush_utils import *


def extract_arr_from_cat(cat, qname, cat_frb=None, cat_galaxy=None):
    '''
    Form an array of quantities out of the (reversed) catalog of gal-FRB pairs.  Will be used in chi^2 calculation.
    qname should be key of cat_frb or cat_galaxy.
    Input either cat_frb or cat_galaxy to extract corresponding quantities.
    '''
    qs = []
    if cat_frb is not None: # extract from FRB catalog
        for ind_frb in cat:
            qs = np.append(qs, cat_frb[qname].values[ind_frb])
    else:
        for ind_frb in cat:
            qs = np.append(qs, cat_galaxy[qname][cat[ind_frb]])
    return qs


class CrossMatchedCat():
    def __init__(self, cat):
        '''
        cat is the dict of gal-FRB pairs.  For chi^2 we need the reversed dict.
        '''
        self.cat = cat
        self.cat_rev = build_reversed_cat(cat)

        # total number of gal-FRB pairs
        self.n_pair = 0
        for ind_gal in cat:
            self.n_pair += len(cat[ind_gal])

    def _get_bounds(self):
        '''
        Called in prep_arrs_for_chi2(); tells the locations of the gal arrays that belong to a FRB.  The array is used in _perturb_FRB_RA_Dec().
        '''
        self.loc_arr = np.zeros(len(self.cat_rev)+1, dtype=int)
        n = 0
        for i, ind_frb in enumerate(self.cat_rev):
            n += len(self.cat_rev[ind_frb])
            self.loc_arr[i+1] = n

    def _stretch_arr(self, vals_):
        '''
        Use the array created by _get_bounds() to stretch a FRB array into the same length as the galaxy arrays.
        '''
        vals = np.zeros(self.n_pair, dtype=vals_.dtype)
        for i in range(len(self.loc_arr)-1):
            vals[self.loc_arr[i]:self.loc_arr[i+1]] = vals_[i]
        return vals

    def prep_arrs_for_chi2(self, cat_galaxy, cat_frb):
        '''
        Call before chi^2 calculation.  This function extracts a lot of arrays from the galaxy and FRB catalogs to facilitate chi^2 calculation.
        '''
        # need to set this up so that we can process the galaxy arrays correctly
        # see _perturb_FRB_RA_Dec()
        self._get_bounds()

        # galaxy arrays
        self.gal_RAs = extract_arr_from_cat(self.cat_rev, 'RAJ2000', cat_galaxy=cat_galaxy)
        self.gal_Decs = extract_arr_from_cat(self.cat_rev, 'DEJ2000', cat_galaxy=cat_galaxy)
        self.gal_dists = extract_arr_from_cat(self.cat_rev, 'Dist', cat_galaxy=cat_galaxy)
        self.Mhalos = extract_arr_from_cat(self.cat_rev, 'Mhalo', cat_galaxy=cat_galaxy)
        # FRB arrays; these should have much shorter lengths
        self.frb_RAs = extract_arr_from_cat(self.cat_rev, 'ra', cat_frb=cat_frb)
        self.frb_Decs = extract_arr_from_cat(self.cat_rev, 'dec', cat_frb=cat_frb)
        self.frb_RA_errs = extract_arr_from_cat(self.cat_rev, 'ra_err', cat_frb=cat_frb)
        self.frb_Dec_errs = extract_arr_from_cat(self.cat_rev, 'dec_err', cat_frb=cat_frb)
        # but the DM array should have the same length as the galaxy arrays
        self.DMs = self._stretch_arr(extract_arr_from_cat(self.cat_rev, 'dm_exc_ne2001', cat_frb=cat_frb))

        # these are the indices of the halo mass... matched to the CGMBrush output
        # used in chi^2 calculation
        self.inds_Mhalo = get_q_inds(self.Mhalos, 'Mhalo')

    def _perturb_FRB_RA_Dec(self):
        '''
        Called in chi^2 calculation.  Randomly perturb the FRB locations according to their localization errors and return the RA Dec values, with the array size the same as those of the galaxy arrays (see prep_arrs_for_chi2()).
        '''
        # get the perturbed RA Dec values first
        frb_RAs = self.frb_RAs + np.random.randn(len(self.frb_RAs))*self.frb_RA_errs
        frb_RAs[frb_RAs<0.] += 360.
        frb_RAs[frb_RAs>360.] -= 360.
        frb_Decs = self.frb_Decs + np.random.randn(len(self.frb_RAs))*self.frb_Dec_errs        
        frb_Decs[frb_Decs<-90.] += 180.
        frb_Decs[frb_Decs>90.] -= 180.

        # Then assign onto arrays with galaxy array length
        frb_RAs = self._stretch_arr(frb_RAs)
        frb_Decs = self._stretch_arr(frb_Decs)

        return frb_RAs, frb_Decs

    def _set_radial_bins(self, bs, b2Rvir_bin_edges):
        '''
        Called in calc_chi2_models() before chi^2 calculation to set up radial bins and the array of bin indicators, to speed up the calculations.
        '''
        b2Rvirs = bs/(0.25*(self.Mhalos/1.3e12)**(1/3))
        iis_b2Rvir_bin = np.zeros((len(b2Rvir_bin_edges)-1,len(self.DMs)), dtype=bool)
        for i in range(len(b2Rvir_bin_edges)-1):
            ii = (b2Rvirs > b2Rvir_bin_edges[i]) & (b2Rvirs < b2Rvir_bin_edges[i+1])
            iis_b2Rvir_bin[i][ii] = True
        return iis_b2Rvir_bin

    def _calc_DMexcs(self, model_data, inds_b):
        '''
        Called in chi^2 calculation.  Calculate the DM excess array by summing up the 1-halo terms and adding up an average 2-halo term.
        '''
        model_DMexcs = np.zeros(len(self.inds_Mhalo))
        model_DMexcs_1halo = model_data[0][(self.inds_Mhalo, inds_b)]
        model_DMexcs_2halo = model_data[1][(self.inds_Mhalo, inds_b)]
        # take care of sightlines passing through multiple halos
        for i in range(len(self.frb_RAs)):
            # add up the 1-halo terms
            DMexc = model_DMexcs_1halo[self.loc_arr[i]:self.loc_arr[i+1]].sum()
            model_DMexcs[self.loc_arr[i]:self.loc_arr[i+1]] = DMexc
            # average the 2-halo terms
            DMexc = np.mean(model_DMexcs_2halo[self.loc_arr[i]:self.loc_arr[i+1]])
            model_DMexcs[self.loc_arr[i]:self.loc_arr[i+1]] += DMexc
        return model_DMexcs

    def _calc_chi2_model(self, model_data, inds_b, alpha, beta, meanDM_all, iis_b2Rvir_bin):
        '''
        Calculate chi^2 for one model.
        '''
        model_DMexcs = self._calc_DMexcs(model_data, inds_b)

        # calculate chi2 in each radial bin
        n_bin = iis_b2Rvir_bin.shape[0]
        chi2s_bin = np.zeros(n_bin)
        n_pairs_bin = np.zeros(n_bin, dtype=int)
        for i in range(n_bin):
            ii = iis_b2Rvir_bin[i]
            n_frb_bin = ii.sum()
            if n_frb_bin == 0:
                continue
            DMs_ = self.DMs[ii] - model_DMexcs[ii]
            # need to remove negative DM FRBs
            DMs_ = DMs_[DMs_ > 0.]
            meanDM_bin = np.average(DMs_, weights=weighting_function(DMs_, alpha, beta))
            chi2s_bin[i] = (meanDM_bin - meanDM_all)**2
            n_pairs_bin[i] = len(DMs_)

        return chi2s_bin, n_pairs_bin

    def calc_chi2_models(self, model_names, alpha, beta, meanDM_all, b2Rvir_bin_edges, marginalize_loc_err=True, N=100000):
        '''
        Calculate chi^2 for one or multiple models.  Can choose to marginalize over localization error or not.
        If marginalize, Monte Carlo N times.
        For the model name(s), only input up to e.g. "fire32_256_2022-04-04".  I need to separate the 1-halo and 2-halo terms.
        '''
        model_datas = load_model_data(model_names)

        if not marginalize_loc_err:
            # evaluate chi^2 using maximum likelihood RA Dec only
            N = 1

        chi2s_bin_ = np.zeros((N, len(model_datas), len(b2Rvir_bin_edges)-1)) # used to store chi^2 for all models
        n_pairs_bin_ = np.zeros((N, len(model_datas), len(b2Rvir_bin_edges)-1), dtype=int) # the number of gal-FRB pairs in each bin
        n_frbs_bin_ = np.zeros((N, len(b2Rvir_bin_edges)-1), dtype=int) # the number of FRBs in each bin
        for i in range(N):
            if i == 0:
                # the first one don't perturb RA Dec
                frb_RAs = self._stretch_arr(self.frb_RAs)
                frb_Decs = self._stretch_arr(self.frb_Decs)
            else:
                # perturb RA Dec
                frb_RAs, frb_Decs = self._perturb_FRB_RA_Dec()
            bs = calc_bs(self.gal_RAs, self.gal_Decs, self.gal_dists, frb_RAs, frb_Decs)
            iis_b2Rvir_bin = self._set_radial_bins(bs, b2Rvir_bin_edges)
            inds_b = get_q_inds(bs, 'b')
            for j in range(len(model_datas)):
                chi2s_bin_[i,j], n_pairs_bin_[i,j] = self._calc_chi2_model(model_datas[j], inds_b, alpha, beta, meanDM_all, iis_b2Rvir_bin)
            for j in range(len(self.frb_RAs)):
                n_frbs_bin_[i] += np.any(iis_b2Rvir_bin[:,self.loc_arr[j]:self.loc_arr[j+1]], axis=1)

        return chi2s_bin_, n_pairs_bin_, n_frbs_bin_

