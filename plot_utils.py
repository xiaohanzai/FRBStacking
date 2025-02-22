import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from FRB_utils import weighting_function, calc_DMexc_distribution


def gen_circle(ra, dec, dist, b):
    '''
    Generate the ra, dec values for a circle of radius b around a galaxy located at ra, dec, dist.
    '''
    c_gal = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    theta = (b / dist * u.rad).to(u.degree)#.value
    cs = c_gal.directional_offset_by(np.linspace(0,360,500)*u.deg, theta)
    return cs


def plot_gal(cat_galaxy, i, b, ax=None, c='b', markersize=5, label=None, linewidth=0.4):
    '''
    Plot galaxy center and circle around it.
    '''
    if ax is None:
        ax = plt.gca()

    # galaxy center
    ax.plot(cat_galaxy[i]['RAJ2000'], cat_galaxy[i]['DEJ2000'], 'x', color=c, markersize=markersize, label=label)

    # circle around the threshold b
    cs = gen_circle(cat_galaxy[i]['RAJ2000'], cat_galaxy[i]['DEJ2000'], cat_galaxy[i]['Dist'], b)
    if np.max(cs.ra)-np.min(cs.ra) < 180*u.deg:
        ax.plot(cs.ra, cs.dec, c, linewidth=linewidth)
    else:
        ii = np.where(cs.ra>180*u.deg)[0]
        ax.plot(cs.ra[:ii[0]], cs.dec[:ii[0]], c, linewidth=linewidth)
        ax.plot(cs.ra[ii], cs.dec[ii], c, linewidth=linewidth)
        ax.plot(cs.ra[ii[-1]+1:], cs.dec[ii[-1]+1:], c, linewidth=linewidth)


def vis_pdf_and_sigmas(DMs, ws, c, n_frb, meanDM=0., ax=None, text=True, label=None, fac=1.):
    '''
    Plot the pdf of the weighted-mean DM (minus their weighted mean), sampling n_frb DM values from the DMs array.
    Also plot the one and two sigma locations.
    c is the color of the lines.
    fac is the correction factor of the width of the distribution because of non-linear weighting.
    '''
    if ax is None:
        ax = plt.gca()

    diff_meanDMs = calc_DMexc_distribution(DMs, ws, n_frb)/fac

    # plot Gaussian
    sigma = np.std(diff_meanDMs)
    mu = meanDM
    if meanDM == 0.:
        x = np.linspace(0, 400, 500)
    else:
        x = np.linspace(mu-3*sigma, mu+3*sigma, 500)
    y = np.exp(-((x-mu)/sigma)**2/2)
    ax.plot(x, y, c, label=label, linewidth=2)

    # plot one and two sigma location
    onesigmas = np.percentile(diff_meanDMs, [16, 84]) + meanDM
    twosigmas = np.percentile(diff_meanDMs, [50-95/2, 50+95/2.]) + meanDM
    ind1 = np.argmin(np.abs(x-onesigmas[0]))
    ind2 = np.argmin(np.abs(x-onesigmas[1]))
    ax.fill_between(x[ind1:ind2], y[ind1:ind2], color=c, alpha=0.5)
    ind1_ = np.argmin(np.abs(x-twosigmas[0]))
    ind2_ = np.argmin(np.abs(x-twosigmas[1]))
    if meanDM != 0.:
        ax.fill_between(x[ind1_:ind1], y[ind1_:ind1], color=c, alpha=0.2)
    ax.fill_between(x[ind2:ind2_], y[ind2:ind2_], color=c, alpha=0.2)
    # label one and two sigma if need to
    if text:
        if meanDM != 0.:
            ax.text(onesigmas[0]-2, y[ind1], r'1-$\sigma$', color=c, fontsize=15,
                horizontalalignment='right', verticalalignment='bottom')
            ax.text(twosigmas[0]-2, y[ind1_], r'2-$\sigma$', color=c, fontsize=15,
                horizontalalignment='right', verticalalignment='bottom')
        else:
            ax.text(onesigmas[1]+2, y[ind2], r'1-$\sigma$', color=c, fontsize=15,
                horizontalalignment='left', verticalalignment='bottom')
            ax.text(twosigmas[1]+2, y[ind2_], r'2-$\sigma$', color=c, fontsize=15,
                horizontalalignment='left', verticalalignment='bottom')

    return np.percentile(diff_meanDMs, [50-95/2,16,50,84,50+95/2])


def vis_ws_and_wavg(DMs, alpha, beta, c, ax=None, ax_in=None):
    '''
    A visualization of the number of 0 weights and whether we suppress high-DM FRBs well enough,
      given the weighting function parameters alpha beta.
    '''
    if ax is None:
        ax = plt.gca()
    ax_in = ax.inset_axes([0.5,0.2,0.45,0.5])

    # make bins in DM
    bin_edges = np.linspace(0,3000,31)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.

    # weights at the bin centers
    ws_bins = weighting_function(bin_centers, alpha, beta)
    ws_bins /= ws_bins.sum()

    # number of DM values in bins
    Ns_bins = np.histogram(DMs, bin_edges)[0].astype(float)
    Ns_bins /= len(DMs)

    # plot number of bins x weight of bin centers
    # this illustrates whether we suppress high-DM FRBs well enough
    ax.plot(bin_centers, Ns_bins*ws_bins, c, linewidth=2)

    # this is the actual mean DM, and the weights
    ws = weighting_function(DMs, alpha, beta)
    meanDM = np.average(DMs, weights=ws)

    # plot mean DM
    ax.plot([meanDM, meanDM], [0, ax.get_ylim()[1]], c, linewidth=2)

    # histogram of the weights
    # we care about number of 0 weights
    ax_in.hist(ws/ws.sum(), color=c, histtype='step', linewidth=2)

    ax.set_xlabel('DM', fontsize=15)
    ax.set_ylabel('# in bin x weight', fontsize=15)


def vis_chi2_pdf(chi2s_bin, labels, ax=None, colors=['tab:blue', 'tab:orange', 'tab:green'], bin_edges=None):
    if ax is None:
        ax = plt.gca()
    ax.hist(chi2s_bin, bins=bin_edges, label=labels, linewidth=2, histtype='step', density=True, color=colors)


def vis_chi2_val(chi2, label, ax=None, ylim=[0, 1], color='k', linestyle='-'):
    if ax is None:
        ax = plt.gca()
    ax.plot([chi2, chi2], ylim, label=label, linewidth=2, color=color, linestyle=linestyle)

