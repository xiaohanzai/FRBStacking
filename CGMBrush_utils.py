import numpy as np
import random
from colossus.cosmology import cosmology
from colossus.lss import mass_function
cosmology.setCosmology('planck18')

base_path = '/Users/xiaohan/Documents/research/FRB_ConnorRavi/codes/mycodes/CGMBrush_models/'

# CGMBrush parameters
model_bs = np.load(base_path+'radial_distance_kpc_res32k.npy')/1e3 # Mpc
model_Mhalos = np.array([1.56841908e+10, 1.89760516e+10, 2.28185389e+10, 2.76090821e+10,
 3.34125995e+10, 4.03236319e+10, 4.87132348e+10, 5.88023286e+10,
 7.10435765e+10, 8.57589215e+10, 1.03574743e+11, 1.25102577e+11,
 1.51112563e+11, 1.82498050e+11, 2.20340026e+11, 2.66143192e+11,
 3.21364686e+11, 3.88210216e+11, 4.68603140e+11, 5.66439120e+11,
 6.83676882e+11, 8.25636629e+11, 9.97696719e+11, 1.20428658e+12,
 1.45500488e+12, 1.75642650e+12, 2.12191894e+12, 2.56597545e+12,
 3.09713234e+12, 3.73259934e+12, 4.51956349e+12, 5.44778410e+12,
 6.58728062e+12, 7.95403273e+12, 9.62743468e+12, 1.16372535e+13,
 1.40173964e+13, 1.69073838e+13, 2.04804358e+13, 2.47073318e+13,
 2.98974026e+13, 3.57863298e+13, 4.35455357e+13, 5.23103045e+13,
 6.35636902e+13, 7.62744827e+13, 9.28231567e+13, 1.11930195e+14,
 1.35614935e+14, 1.60876513e+14, 1.95131274e+14, 2.31709184e+14,
 2.83507935e+14, 3.41714289e+14, 4.20401787e+14, 4.92095239e+14,
 0.00000000e+00, 6.80999996e+14, 0.00000000e+00, 1.07742855e+15])
model_Rvirs = np.array([0.06546749, 0.06976007, 0.0741824,  0.07904765, 0.08423807, 0.08968604,
 0.09551845, 0.10170363, 0.10832114, 0.11533596, 0.1228258,  0.13080581,
 0.13930664, 0.14835119, 0.15796811, 0.16823257, 0.17914486, 0.1907922,
 0.20314526, 0.2163994,  0.23040325, 0.2453586,  0.26133915, 0.27825838,
 0.29636468, 0.31555977, 0.33608404, 0.35805999, 0.38123394, 0.4057035,
 0.43241803, 0.46019821, 0.49027568, 0.52207643, 0.55638467, 0.59268208,
 0.63060985, 0.67126979, 0.71556874, 0.76175196, 0.81173845, 0.86187447,
 0.92013774, 0.97813931, 1.04377766, 1.10916985, 1.18419642, 1.26043577,
 1.34371595, 1.42244619, 1.51698257, 1.60639637, 1.71814589, 1.82849037,
 1.95926486, 2.06484768, 0.,         2.30102145, 0.,         2.68123239]) # Mpc

# used for generating mock data
model_Mhalo_weights = mass_function.massFunction(model_Mhalos.clip(1e9)*0.67, 0, q_in='M', q_out='dndlnM', model = 'sheth99')
model_Mhalo_weights[-2] = model_Mhalo_weights[-4] = 0


def gen_mock_data(n_frb, model_name, mockDMs=None, Mhalo_min=None, Mhalo_max=None, nRvir=2., min_nRvir=0.25):
    '''
    Create n_frb mock FRBs.  If mockDMs is given, select base DM from this array.  Otherwise base DM is 0.
    Randomly choose halo masses within the range [Mhalo_min, Mhalo_max], with weights given by the halo mass function, the model_Mhalo_weights array.
    Randomly choose impact parameters (b2Rvir ratio) within the range [min_nRvir, nRvir].
    Assign DM excess to mock FRBs based on the halo masses and impact parameters.
    '''
    # the base DM
    DMs = np.zeros(n_frb)
    if mockDMs is not None:
        inds = np.random.choice(len(mockDMs), n_frb, replace=True)
        DMs = mockDMs[inds]*1.
    # randomly assign halo masses
    tmp = np.linspace(0, len(model_Mhalos)-1, len(model_Mhalos), dtype=int)
    if Mhalo_min and Mhalo_max:
        #print(True)
        ii = (model_Mhalos > Mhalo_min) & (model_Mhalos < Mhalo_max)
        tmp = tmp[ii]
        tmp_weights = model_Mhalo_weights[ii]
    else:
        ii = model_Mhalos > 1.
        tmp = tmp[ii]
        tmp_weights = None
    inds = random.choices(tmp, weights=tmp_weights, k=n_frb)
    Mhalos = model_Mhalos[inds]
    Rvirs = model_Rvirs[inds]
    # randomly generate impact parameters
    bs = Rvirs*np.random.uniform(min_nRvir, nRvir, n_frb)
    # get excess DM and add to the base DM
    DMexcs = DMs*0.
    if type(model_name) == str:
        model_DMexcs = np.load(base_path+model_name)[inds]
        for i in range(n_frb):
            DMexcs[i] = model_DMexcs[i][np.argmin(np.abs(bs[i] - model_bs))]
    else:
        DMexcs += model_name # const DM excess; input a number
    for i in range(n_frb):
        DMs[i] += DMexcs[i]
    return Mhalos, bs, DMs, DMexcs # convert b to Mpc


def get_q_inds(qs, qname):
    '''
    Get the nearest indicies of an array of halo masses or impact parameters.  qname should indicate which quantity.
    '''
    inds = np.zeros(len(qs), dtype=int)
    for i in range(len(qs)):
        if qname == 'Mhalo':
            ind = np.argmin(np.abs(np.log10(qs[i]) - np.log10(model_Mhalos.clip(1e-6))))
        else: # b; we won't get out of the max model_bs
            ind = np.argmin(np.abs(qs[i] - model_bs))
        inds[i] = ind
    return inds


def get_model_DMexc(Mhalos=None, bs=None, model_name=0., inds_Mhalo=None, inds_b=None):
    '''
    Input either the Mhalos and bs, or the indices of Mhalos and bs, find the model predicted DM excess.
    model_name is name of the model to be evaluated.  If input a number, interpret as const DM excess.
    '''
    if type(model_name) == str:
        if inds_Mhalo is None:
            inds_Mhalo = get_q_inds(Mhalos, 'Mhalo')
            inds_b = get_q_inds(bs, 'b')
        model_DMexcs = np.load(base_path+model_name)[(inds_Mhalo, inds_b)] - 81. # mean DM is 81 in CGMBrush
    else:
        tmp = Mhalos
        if Mhalos is None:
            tmp = inds_Mhalo
        model_DMexcs = tmp*0. + model_name # const DM excess; input a number
    return model_DMexcs


def load_model_data(model_names):
    '''
    Model name(s) should be input up to e.g. "fire32_256_2022-04-04".  Will separate the one and two-halo terms and return both.
    '''
    if type(model_names) is not list:
        model_names = [model_names]

    model_datas = [None]*len(model_names)
    for i in range(len(model_names)):
        model_name = model_names[i]
        if type(model_name) == str:
            model_1halo = np.load(base_path+model_name+'_masks.npy')
            model_2halo = np.load(base_path+model_name+'_DMvsR_prof.npy') - model_1halo - 81. # mean DM is 81 in CGMBrush
        else: # input a number; const DM excess; no 2-halo term?
            model_1halo = np.zeros((len(model_Mhalos), len(model_bs))) + model_name
            model_2halo = model_1halo*0.
        model_datas[i] = [model_1halo, model_2halo]

    return model_datas

