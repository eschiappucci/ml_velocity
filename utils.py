import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from colossus.cosmology import cosmology
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split as array_split
from astropy.io import fits

params = {'flat': True, 'H0': 67.66, 'Om0': (0.11933+0.02242)/(0.6766**2), 'Ob0': 0.02242/(0.6766**2), 'sigma8': 0.8102, 'ns': 0.9665}
cosmo = cosmology.setCosmology('FullPlanck18', params)

h = cosmo.Hz(0)/100
print(cosmo.name)

RA_SPTSZ_min = -50
RA_SPTSZ_max = 50
DEC_SPTSZ_min = -70
DEC_SPTSZ_max = -40

def fn_load_halo(fname, Mmin=1e14, Mmax=1e16, zmin=0.1 , zmax=1., nobj=None, like_SPTSZ=False):
    """
    Returns a RA, DEC, Z with redmapper catalog info.
    """
    cat = fits.open(fname)[1].data
    cat = cat[(cat.REDSHIFT >= zmin) & (cat.REDSHIFT <= zmax)]
    cat = cat[(cat.M500 >= Mmin) & (cat.M500 <= Mmax)]
    if like_SPTSZ:
        cat = cat[(cat.RA >= RA_SPTSZ_min) & (cat.RA <= RA_SPTSZ_max)]
        cat = cat[(cat.DEC >= DEC_SPTSZ_min) & (cat.DEC <= DEC_SPTSZ_max)]

    if nobj is not None:
        cat = cat[np.random.randint(0,len(cat),size=nobj)]

    cat.RA[np.where(cat.RA > 180)] = cat.RA[np.where(cat.RA > 180)] - 360
    ra    = cat.RA
    dec   = cat.DEC
    zs    = cat.REDSHIFT
    v_los = cat.VLOS
    M200  = cat.M200

    return ra, dec, zs, v_los, M200

def readdata(filename, DECmin=None, DECmax=None, Zmin=None, Zmax=None, richmin=None, richmax=None, photoz=None, sigmaM=None, nobj=None, seed=None):
    clustinfo = ascii.read(filename)
    #print(clustinfo.keys())
    Z, TSZ, richness = np.asarray(clustinfo['Z']),np.asarray(clustinfo['TSZ']),np.asarray(clustinfo['M200'])
    RA = np.asarray(clustinfo['RA'])
    DEC = np.asarray(clustinfo['DEC'])
    TSZ = -TSZ
    vlos = np.asarray(clustinfo['vlos'])

    if DECmin is not None and DECmax is not None:
        pos = np.concatenate((np.where(clustinfo['DEC'] > DECmax)[0] , np.where(clustinfo['DEC'] < DECmin)[0]))
        Z = np.delete(Z, pos)
        TSZ = np.delete(TSZ, pos)
        richness = np.delete(richness, pos)
        RA = np.delete(RA, pos)
        DEC = np.delete(DEC, pos)
        vlos = np.delete(vlos, pos)

    if sigmaM is not None:
        if seed is not None:
            np.random.seed(seed)
        Merr = np.random.normal(loc=0., scale=sigmaM)
        richness = np.exp(np.log(richness) + Merr)

    if richmin is not None:
        pos = np.where(richness < richmin)
        Z = np.delete(Z, pos)
        TSZ = np.delete(TSZ, pos)
        richness = np.delete(richness, pos)
        RA = np.delete(RA, pos)
        DEC = np.delete(DEC, pos)
        vlos = np.delete(vlos, pos)
    if richmax is not None:
        pos = np.where(richness > richmax)
        Z = np.delete(Z, pos)
        TSZ = np.delete(TSZ, pos)
        richness = np.delete(richness, pos)
        RA = np.delete(RA, pos)
        DEC = np.delete(DEC, pos)
        vlos = np.delete(vlos, pos)
    if nobj is not None:
        if seed is not None:
            np.random.seed(seed)
        pos = np.arange(len(Z))
        np.random.shuffle(pos)
        pos = pos[0:nobj]
        Z = Z[pos]
        RA = RA[pos]
        DEC = DEC[pos]
        TSZ = TSZ[pos]
        richness = richness[pos]
        vlos = vlos[pos]

    Zerr = 0
    if photoz is not None:
        if seed is not None:
            np.random.seed(seed)
        Zerr = np.random.normal(loc=0., scale=photoz*(1+Z))
        Z += Zerr

    com_dists = cosmo.comovingDistance(z_max=Z)/h          # Compute comoving distance to each cluster
    return RA, DEC, com_dists, richness, TSZ, Z, vlos

def RaDec2XYZ(ra,dec):
    """
    From (ra,dec) -> unit vector on the sphere
    """
    rar  = np.radians(ra)
    decr = np.radians(dec)

    x = np.cos(rar) * np.cos(decr)
    y = np.sin(rar) * np.cos(decr)
    z = np.sin(decr)

    return np.array([x,y,z]).T

def build_tf_matrix_vec(ra, dec, z, mass, vlos, close_pairs=4):
    vec_unit = RaDec2XYZ(ra,dec) # Get unit vectors pointing to the cluster
    n_clusts = len(ra)
    vec_dist = (vec_unit.T * z).T # Mpc
    tree = cKDTree(vec_dist)
    dists, ind = tree.query(vec_dist, k=close_pairs+1)
    tf_mat = np.zeros((n_clusts, close_pairs+1, 5))
    for i in range(n_clusts):
        for j, v in enumerate(ind[i]):
            tf_mat[i,j] = np.array([ra[v]/180, dec[v]/90, z[v], mass[v]/1e15, dists[i,j]])

    return tf_mat, np.exp(vlos/1e3)

def tf_regression(mat, output_shape = 1, rate=1e-5, loss_func='mse', metric='mean_absolute_percentage_error'):
    model = keras.Sequential([
        keras.layers.ZeroPadding1D(2, input_shape=mat[0].shape),
        keras.layers.LocallyConnected1D(128, 4, data_format='channels_last'),
        keras.layers.LocallyConnected1D(64, 5, data_format='channels_last'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(512, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='elu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(output_shape, activation='exponential')
        ])
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
    return model

def plot_metric(history, metric='loss'):
    plt.plot(history.history[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Change of '+metric)
    plt.legend()
    plt.show()

def recompile(loc, rate=1e-5, loss_func='mse', metric='mean_absolute_percentage_error'):
    model = tf.keras.models.load_model(loc, compile=False)
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=[metric])
    return model


