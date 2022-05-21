import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def build_gal_FRB_pairs(cat_galaxy, ii_halo, cat_frb, b_thres, build_cat=False):
    '''
    Build galaxy-FRB pairs by finding FRBs with b < b_thres, where b_thres can be an array.
    ii_halo determines which galaxies to take into consideration.
    If build_cat, build a dict where keys = galaxy indices, values = FRB indices.
    '''
    c_gals = SkyCoord(ra=cat_galaxy['RAJ2000'][ii_halo]*u.degree, dec=cat_galaxy['DEJ2000'][ii_halo]*u.degree)

    cat = None
    if build_cat:
        cat = {} # key is indices of galaxies, value is indices of FRBs
    ii_frb = np.zeros(len(cat_frb), dtype=bool)
    inds_gal = np.linspace(0, len(cat_galaxy)-1, len(cat_galaxy), dtype=int)[ii_halo]
    for i in range(len(cat_frb)):
        c_frb = SkyCoord(ra=cat_frb['ra'].values[i]*u.degree, dec=cat_frb['dec'].values[i]*u.degree)
        thetas = c_frb.separation(c_gals)
        ii = (cat_galaxy['Dist'][ii_halo]*np.sin(thetas) < b_thres) & (thetas < 90.*u.degree)
        if ii.sum()>0:
            ii_frb[i] = True
            if build_cat:
                for ind in inds_gal[ii]:
                    if ind not in cat:
                        cat[ind] = [i]
                    else:
                        cat[ind].append(i)

    return ii_frb, cat


def get_subcat(cat, ii_frb, cat_galaxy, col_name, col_min, col_max):
    '''
    Given a dict of gal-FRB pairs, return the sub-catalog where the column of cat_galaxy is bounded by col_min, col_max.
    '''
    cat_ = {}
    ii_frb_ = np.zeros_like(ii_frb)
    for i in cat:
        col_val = cat_galaxy[col_name][i]
        if col_val > col_min and col_val < col_max:
            cat_[i] = cat[i]
            ii_frb_[cat[i]] = True
    return cat_, ii_frb_


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


def extract_arr_from_cat(cat, qname, cat_frb=None, cat_galaxy=None):
    '''
    Form an array of quantities out of the catalog of gal-FRB pairs.  Will be used in chi^2 calculation.
    qname should be key of cat_frb or cat_galaxy.
    Input either cat_frb or cat_galaxy to extract corresponding quantities.
    '''
    qs = []
    if cat_frb is not None: # extract from FRB catalog
        for ind_gal in cat:
            qs = np.append(qs, cat_frb[qname].values[cat[ind_gal]])
    else:
        for ind_gal in cat:
            qs = np.append(qs, [cat_galaxy[qname][ind_gal]]*len(cat[ind_gal]))
    return qs


def calc_bs(gal_RAs, gal_Decs, gal_dists, frb_RAs, frb_Decs):
    '''
    Given arrays of galaxy RA Dec distance Rvir, and FRB RA Dec, calculate the impact parameter normalized to the virial radii.  Better used after extract_arr_from_cat().
    '''
    c_frbs = SkyCoord(ra=frb_RAs*u.degree, dec=frb_Decs*u.degree)
    c_gals = SkyCoord(ra=gal_RAs*u.degree, dec=gal_Decs*u.degree)
    thetas = c_gals.separation(c_frbs)
    bs = np.sin(thetas)*gal_dists
    return bs

