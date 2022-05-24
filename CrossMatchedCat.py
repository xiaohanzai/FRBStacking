import numpy as np
import scipy.stats
import random
from gal_utils import *
from FRB_utils import *
from crossmatch_utils import *
from plot_utils import *
from chi2_CGMBrush import *
from CGMBrush_utils import *


class CrossMatchedCat():
    def __init__(self, cat):
        '''
        cat is the dict of gal-FRB pairs.
        '''
        self.cat = cat

    def prep_arrs_for_chi2(self, cat_galaxy, cat_frb):
        '''
        Call before chi^2 calculation.  This function extracts a lot of arrays from the galaxy and FRB catalogs to facilitate chi^2 calculation.
        '''
        self.gal_RAs = extract_arr_from_cat(self.cat, 'RAJ2000', cat_galaxy=cat_galaxy)
        self.gal_Decs = extract_arr_from_cat(self.cat, 'DEJ2000', cat_galaxy=cat_galaxy)
        self.gal_dists = extract_arr_from_cat(self.cat, 'Dist', cat_galaxy=cat_galaxy)
        self.frb_RAs = extract_arr_from_cat(self.cat, 'ra', cat_frb=cat_frb)
        self.frb_Decs = extract_arr_from_cat(self.cat, 'dec', cat_frb=cat_frb)
        self.frb_RA_errs = extract_arr_from_cat(self.cat, 'ra_err', cat_frb=cat_frb)
        self.frb_Dec_errs = extract_arr_from_cat(self.cat, 'dec_err', cat_frb=cat_frb)
        self.DMs = extract_arr_from_cat(self.cat, 'dm_exc_ne2001', cat_frb=cat_frb)
        self.Mhalos = extract_arr_from_cat(self.cat, 'Mhalo', cat_galaxy=cat_galaxy)
        #self.gal_Rvirs = (self.Mhalos/1.3e12)**(1/3)*0.25

        self.inds_Mhalo = get_q_inds(self.Mhalos, 'Mhalo')

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

    def _calc_chi2_model(self, model_name, inds_b, alpha, beta, meanDM_all, iis_b2Rvir_bin):
        '''
        Calculate chi^2 for one model.
        '''
        model_DMexcs = get_model_DMexc(model_name=model_name, inds_Mhalo=self.inds_Mhalo, inds_b=inds_b)

        # calculate chi2 in each radial bin
        n_bin = iis_b2Rvir_bin.shape[0]
        chi2s_bin = np.zeros(n_bin)
        for i in range(n_bin):
            ii = iis_b2Rvir_bin[i]
            n_frb_bin = ii.sum()
            if n_frb_bin == 0:
                continue
            DMs_ = self.DMs[ii] - model_DMexcs[ii]
            meanDM_bin = np.average(DMs_, weights=weighting_function(DMs_, alpha, beta))
            chi2s_bin[i] = (meanDM_bin - meanDM_all)**2

        return chi2s_bin

    def calc_chi2_models(self, model_names, alpha, beta, meanDM_all, b2Rvir_bin_edges, marginalize_loc_err=True, N=100000):
        '''
        Calculate chi^2 for one or multiple models.  Can choose to marginalize over localization error or not.
        If marginalize, Monte Carlo N times.
        '''
        if type(model_names) is not list:
            model_names = [model_names]

        if not marginalize_loc_err:
            # evaluate chi^2 using maximum likelihood RA Dec only
            N = 1

        chi2s_bin_ = np.zeros((N, len(model_names), len(b2Rvir_bin_edges)-1)) # used to store chi^2 for all models
        n_frbs_bin_ = np.zeros((N, len(b2Rvir_bin_edges)-1)) # used to store chi^2 for all models
        for i in range(N):
            if i == 0:
                # the first one don't perturb RA Dec
                frb_RAs = self.frb_RAs
                frb_Decs = self.frb_Decs
            else:
                # perturb RA Dec
                frb_RAs = self.frb_RAs+np.random.randn(len(self.frb_RAs))*self.frb_RA_errs
                frb_RAs[frb_RAs<0.] += 360.
                frb_RAs[frb_RAs>360.] -= 360.
                frb_Decs = self.frb_Decs+np.random.randn(len(self.frb_RAs))*self.frb_Dec_errs
                frb_Decs[frb_Decs<-90.] += 180.
                frb_Decs[frb_Decs>90.] -= 180.
            bs = calc_bs(self.gal_RAs, self.gal_Decs, self.gal_dists, frb_RAs, frb_Decs)
            # print(bs)
            iis_b2Rvir_bin = self._set_radial_bins(bs, b2Rvir_bin_edges)
            inds_b = get_q_inds(bs, 'b')
            for j in range(len(model_names)):
                chi2s_bin_[i,j] = self._calc_chi2_model(model_names[j], inds_b, alpha, beta, meanDM_all, iis_b2Rvir_bin)
            n_frbs_bin_[i] = iis_b2Rvir_bin.sum(axis=1)

        return chi2s_bin_, n_frbs_bin_

