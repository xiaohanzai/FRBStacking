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


def find_groups(cat_galaxy):
    '''
    Identify galaxy groups and remove the non-central galaxies from the catalog.
    '''
    Rvirs = calc_Rvir(cat_galaxy['Mhalo'])
    c_gals = SkyCoord(ra=cat_galaxy['RAJ2000']*u.degree, dec=cat_galaxy['DEJ2000']*u.degree,
              distance=cat_galaxy['Dist']*u.Mpc)
    i = 0
    while i < len(cat_galaxy)-1:
        c_gal = c_gals[i]
        # within 1 Rvir... and some percentage
        inds_in = np.where(c_gal.separation_3d(c_gals).value < 1.2*Rvirs)[0]
        if len(inds_in) == 1:
            i += 1
            continue
        # set the most massive one as the central galaxy
        ind_central = inds_in[np.argmax(Rvirs[inds_in])]
        inds_in = inds_in[inds_in != ind_central]
        cat_galaxy.remove_rows(inds_in)
        Rvirs = np.delete(Rvirs, inds_in)
        c_gals = np.delete(c_gals, inds_in)
        if ind_central == i:
            i += 1


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
    # # M32 not considered(?)
    # Mhalos[inds_gal == 885] = 0 # M32 should be removed in the find_groups() step

    # put things into a new table
    cat_galaxy = Table()
    for key in cat_gal.columns.names:
        cat_galaxy[key] = cat_gal[key]
    cat_galaxy['Mhalo'] = Mhalos
    cat_galaxy['Mstar'] = 10**logMstars

    # get rid of too massive ones -- problematic
    cat_galaxy = cat_galaxy[cat_galaxy['Mhalo']<5e15]

    # known corrections... refer to GLADE+, even though they seem wrong
    # I idenditied >1e13 galaxies that intersect FRBs and googled a bunch of them;
    # NGC2256 and 2258 are in the MASSIVE survey
    # NGC0741 (or IC1751?) and 1961 indeed seem very massive
    for pair in [['PGC086434', 1e10],
                 ['NGC1599', 2e12],
                 ['NGC1600', 1.01e12],
                 ['IC0678', 1e10],
                 ['NGC4377', 2.3e15]]:
        name, Mhalo = pair
        cat_galaxy['Mhalo'][cat_galaxy['Name'] == name] = Mhalo

    # take care of M33 when removing satellites of groups...
    tmp = cat_galaxy[cat_galaxy['Name'] == 'NGC0598'].values()
    find_groups(cat_galaxy)
    if 'NGC0598' not in cat_galaxy['Name']:
        cat_galaxy.add_row(tmp)

    return cat_galaxy

