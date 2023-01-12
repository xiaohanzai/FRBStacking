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


def find_groups(cat_galaxy, method):
    '''
    Identify galaxy groups and remove the non-central galaxies from the catalog.
    '''
    Rvirs = calc_Rvir(cat_galaxy['Mhalo']) # Mpc
    vcs = (4.3e-9*cat_galaxy['Mhalo']/Rvirs)**0.5 # km/s
    c_gals = SkyCoord(ra=cat_galaxy['RAJ2000']*u.degree, dec=cat_galaxy['DEJ2000']*u.degree,
              distance=cat_galaxy['Dist']*u.Mpc)
    i = 0
    while i < len(cat_galaxy)-1:
        c_gal = c_gals[i]
        # find all galaxies more massive than this one
        inds = np.where(Rvirs > Rvirs[i])[0]
        if method == '3d distance':
            inds_in = np.where(c_gal.separation_3d(c_gals[inds]).value < 1.2*Rvirs[inds])[0]
        else:
            # 2D distance within 1 Rvir... and some percentage
            thetas = c_gal.separation(c_gals[inds])
            if method == '2d+distance':
                # consider radial distances error
                ii = np.abs(cat_galaxy['Dist'][inds] - cat_galaxy['Dist'][i]) < \
                    np.clip(cat_galaxy['e_Dist'][inds], cat_galaxy['e_Dist'][i], None)
            else:
                # velocity
                ii = np.abs(cat_galaxy['RadialVelocity'][inds] - cat_galaxy['RadialVelocity'][i]) < \
                    3*np.clip(vcs[inds], vcs[i], None)
            inds_in = np.where(
                (cat_galaxy['Dist'][inds]*np.sin(thetas) < 1.2*Rvirs[inds]) & (thetas < 90.*u.degree) & ii)[0]
        if len(inds_in) == 0:
            i += 1
            continue
        # remove this galaxy
        cat_galaxy.remove_row(i)
        Rvirs = np.delete(Rvirs, i)
        vcs = np.delete(vcs, i)
        c_gals = np.delete(c_gals, i)


def load_gals(load_all=False, method='3d distance'):
    '''
    Load in galaxy catalog.
    '''
    if load_all:
        data = np.loadtxt('gwgc_Mstar.txt')
        # indices
        inds_gal = data[:,0].astype(int)
        # get data
        with fits.open('gwgc_binary.fit') as f:
            cat_gal = f[1].data[inds_gal]
        # masses
        logMstars = np.log10(data[:,1])
        Mhalos = 10**np.array([logMhalofromlogMstar(i) for i in logMstars])
        # known corrections
        for pair in [['PGC086434', 1e10],
                     ['NGC1599', 2e12],
                     ['NGC1600', 1.01e12],
                     ['IC0678', 1e10],
                     ['NGC4377', 2.3e15]]:
            name, Mhalo = pair
            Mhalos[cat_gal['Name'] == name] = Mhalo
        # make table
        cat_galaxy = Table()
        for key in cat_gal.columns.names:
            cat_galaxy[key] = cat_gal[key]
        cat_galaxy['Mhalo'] = Mhalos
        cat_galaxy['Mstar'] = 10**logMstars
        # remove too massive ones
        cat_galaxy = cat_galaxy[cat_galaxy['Mhalo']<5e15]
    else:
        # see galaxy_catalog.ipynb for pre-processing
        cat_galaxy = Table.read('cat_galaxy_>1e11.fits')

    if method is not None:
        # don't do group finding if method is None
        # find groups and remove satellites
        # take care of M33 when removing satellites of groups...
        tmp = cat_galaxy[cat_galaxy['Name'] == 'NGC0598'].values()
        find_groups(cat_galaxy, method)
        if 'NGC0598' not in cat_galaxy['Name']:
            cat_galaxy.add_row(tmp)

    return cat_galaxy

