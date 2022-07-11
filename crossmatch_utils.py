import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def build_gal_FRB_pairs(cat_galaxy, ii_halo, cat_frb, b_thres, b_thres_lo=None, build_cat=False):
    '''
    Build galaxy-FRB pairs by finding FRBs with b < b_thres, where b_thres can be an array.
    If b_thres_lo is provided, do b_thres_lo < b < b_thres.
    ii_halo determines which galaxies to take into consideration.
    If build_cat, build a dict where keys = galaxy indices, values = FRB indices.
    '''
    c_gals = SkyCoord(ra=cat_galaxy['RAJ2000'][ii_halo]*u.degree, dec=cat_galaxy['DEJ2000'][ii_halo]*u.degree)

    if b_thres_lo is None:
        b_thres_lo = b_thres*0.

    cat = None
    if build_cat:
        cat = {} # key is indices of galaxies, value is indices of FRBs
    ii_frb = np.zeros(len(cat_frb), dtype=bool)
    inds_gal = np.linspace(0, len(cat_galaxy)-1, len(cat_galaxy), dtype=int)[ii_halo]
    for i in range(len(cat_frb)):
        c_frb = SkyCoord(ra=cat_frb['ra'].values[i]*u.degree, dec=cat_frb['dec'].values[i]*u.degree)
        thetas = c_frb.separation(c_gals)
        ii = (cat_galaxy['Dist'][ii_halo]*np.sin(thetas) < b_thres) & (cat_galaxy['Dist'][ii_halo]*np.sin(thetas) > b_thres_lo) & (thetas < 90.*u.degree)
        if ii.sum()>0:
            ii_frb[i] = True
            if build_cat:
                for ind in inds_gal[ii]:
                    if ind not in cat:
                        cat[ind] = [i]
                    else:
                        cat[ind].append(i)

    return ii_frb, cat


def build_reversed_cat(cat):
    '''
    The dict returned by build_gal_FRB_pairs() uses gal indices as keys and FRB indices as values.  When doing chi^2 calculation I'm going to need the FRBs as keys and galaxies as values.  This function creates the reversed dict given the gal-FRB dict.
    '''
    cat_rev = {}
    for ind_gal in cat:
        for ind_frb in cat[ind_gal]:
            if ind_frb in cat_rev:
                cat_rev[ind_frb].append(ind_gal)
            else:
                cat_rev[ind_frb] = [ind_gal]
    return cat_rev


def is_subcat_criteria_met(ind_gal, cat_galaxy, subcat_criteria):
    '''
    Determine whether a galaxy meets the criteria to be put in sub-catalog.
    '''
    flag = True
    for col_name in subcat_criteria:
        col_val = cat_galaxy[col_name][ind_gal]
        col_min, col_max = subcat_criteria[col_name]
        if col_val < col_min or col_val > col_max:
            flag = False
            break
    return flag


def get_subcat(cat, ii_frb, cat_galaxy, subcat_criteria):
    '''
    Given a dict of gal-FRB pairs, return the sub-catalog where the column of cat_galaxy is bounded by col_min, col_max.
    '''
    cat_ = {}
    ii_frb_ = np.zeros_like(ii_frb)
    for ind_gal in cat:
        if is_subcat_criteria_met(ind_gal, cat_galaxy, subcat_criteria):
            cat_[ind_gal] = cat[ind_gal]
            ii_frb_[cat[ind_gal]] = True
    return cat_, ii_frb_


def get_subcat_reversed(cat, cat_galaxy, subcat_criteria):
    '''
    Build reversed sub-catalog.  Key is FRB index, value is list of **all** galaxy indices.
    FRBs only include those of the subcat, e.g. bounded by halo mass.
    This is used in chi^2 calculation because we need to calculate DM excess contributed by all halos.
    '''
    # get the indices of FRBs of the subcat first
    inds_frb = [0] # just to make sure dtype is int
    for ind_gal in cat:
        if is_subcat_criteria_met(ind_gal, cat_galaxy, subcat_criteria):
            inds_frb = np.append(inds_frb, cat[ind_gal])
    inds_frb = inds_frb[1:]

    # build reversed subcat
    # initialize
    cat_rev = {}
    for ind_frb in inds_frb:
        cat_rev[ind_frb] = []
    # add in galaxy indices
    for ind_gal in cat:
        for ind_frb in cat[ind_gal]:
            if ind_frb in inds_frb:
                cat_rev[ind_frb].append(ind_gal)
    return cat_rev


def calc_bs_1gal(cat_frb, inds_frb, cat_galaxy, ind_gal, divide_Rvir=True):
    '''
    Given galaxy ind_gal and the FRB indices that intersect it, calculate impact parameters.
    '''
    c_frbs = SkyCoord(ra=cat_frb['ra'].values[inds_frb]*u.degree, dec=cat_frb['dec'].values[inds_frb]*u.degree)
    c_gal = SkyCoord(ra=cat_galaxy['RAJ2000'][ind_gal]*u.degree, dec=cat_galaxy['DEJ2000'][ind_gal]*u.degree)
    thetas = c_gal.separation(c_frbs)
    bs = np.sin(thetas)*cat_galaxy['Dist'][ind_gal]
    if divide_Rvir:
        Rvir = 0.25*(cat_galaxy['Mhalo'][ind_gal]/1.3e12)**(1/3)
        bs /= Rvir
    return bs


def calc_bs(gal_RAs, gal_Decs, gal_dists, frb_RAs, frb_Decs):
    '''
    Given arrays of galaxy RA Dec distance Rvir, and FRB RA Dec, calculate the impact parameter normalized to the virial radii.  Better used after extract_arr_from_cat().
    '''
    c_frbs = SkyCoord(ra=frb_RAs*u.degree, dec=frb_Decs*u.degree)
    c_gals = SkyCoord(ra=gal_RAs*u.degree, dec=gal_Decs*u.degree)
    thetas = c_gals.separation(c_frbs)
    bs = np.sin(thetas)*gal_dists
    return bs

