import numpy as np
from scipy.optimize import fsolve
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table


# halo virial radius based on halo mass
calc_Rvir = lambda Mhalo: 0.25*(Mhalo/1.3e12)**(1/3)


#Moster et al abundance matching relation: http://iopscience.iop.org/article/10.1088/0004-637X/710/2/903/pdf
def logMstarfromlogMhalo(logMhalo):
    mM0 = 0.02820
    beta = 1.057
    gamma = 0.556
    logM1 = 11.884
    return logMhalo+ np.log10(2.*mM0*((10**(logMhalo-logM1))**-beta + (10**(logMhalo-logM1))**gamma)**-1)

#inverts the Moster relation to give halo mass as a function of stellar mass
def logMhalofromlogMstar(logMstar):
    sub = lambda logMhalo: logMstarfromlogMhalo(logMhalo) - logMstar
    return fsolve(sub, 12)[0] #center on 10^12 Msun halos


def load_gals():
    '''
    Load in galaxy catalog, indices to be used, and calculate halo masses.
    The returned galaxy catalog does not contain the unused galaxies so indices differ from the original.
    But inds_gal can be used to match things.
    '''
    # indices of galaxies that we are interested in
    inds_gal = np.loadtxt('gwgc_Mstar.txt')[:,0].astype(int)

    # load in galaxy catalog
    with fits.open('gwgc_binary.fit') as f:
        cat_gal = f[1].data[inds_gal]

    # get halo masses
    logMstars = np.log10(np.loadtxt('gwgc_Mstar.txt')[:,1])
    Mhalos = 10**np.array([logMhalofromlogMstar(i) for i in logMstars])
    # M32 not considered(?)
    Mhalos[inds_gal == 885] = 0

    # probably need to exclude Virgo cluster and every galaxy inside one virial radius
    # also those with too large halo mass
    c_gals = SkyCoord(ra=cat_gal['RAJ2000']*u.degree,
                  dec=cat_gal['DEJ2000']*u.degree)
    ind = np.where(cat_gal['name'] == 'NGC4377')[0]
    c_gal = c_gals[ind]
    thetas = c_gal.separation(c_gals)
    ii_virgo = (cat_gal[ind]['Dist']*np.sin(thetas) < 3.) & (thetas < 90.*u.degree)

    # put things into a new table
    cat_galaxy = Table()
    for key in cat_gal.columns.names:
        cat_galaxy[key] = cat_gal[key]
    cat_galaxy['Mhalo'] = Mhalos
    cat_galaxy['Mstar'] = 10**logMstars

    return cat_galaxy, inds_gal, ii_virgo

